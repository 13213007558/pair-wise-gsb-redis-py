"""Offline regression tests for the tracking connection / cache-entry lifecycle.

These tests pin the rules that keep a client-side cached reply trustworthy when the
connection that obtained it changes state. They run entirely over an in-process
``socket.socketpair`` fake server - no live Redis - so the regression chain can be
exercised in CI without a server, including the failure branches a real server is
awkward to drive into.

The real implementation ties a cached reply to a tracking session through three links,
and every test below exercises one of them:

- ``CacheEntry.connection_ref`` names the wrapped connection that fetched the reply. A
  later hit drains that connection first, so invalidations parked on an idle owner are
  applied before the entry can be served.
- ``CacheProxyConnection`` enables tracking from a connect callback
  (``_enable_tracking_callback``), so every fresh socket re-runs the
  ``CLIENT TRACKING ON`` handshake and re-binds the parser's invalidation handler.
- The server destroys a connection's tracking state when the socket drops. Any entry
  cached through the old session therefore has no invalidation channel left and must be
  evicted on the connection lifecycle change, not kept as if still tracked.

What keeps a valid notification channel (and so may stay cached) versus what must be
evicted is the subject of each grouped class below.
"""

import socket
import threading

import pytest
import redis
from redis.backoff import NoBackoff
from redis.cache import (
    CacheConfig,
    CacheEntry,
    CacheEntryStatus,
    CacheKey,
    DefaultCache,
    TrackingMode,
)
from redis.connection import CacheProxyConnection, Connection, _NO_PENDING_HIT
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError, ResponseError
from redis.retry import Retry


def _bulk(value):
    if value is None:
        return b"$-1\r\n"
    if isinstance(value, str):
        value = value.encode()
    return b"$" + str(len(value)).encode() + b"\r\n" + value + b"\r\n"


def _array(items):
    return (
        b"*"
        + str(len(items)).encode()
        + b"\r\n"
        + b"".join(_bulk(item) for item in items)
    )


def _map(mapping):
    frame = b"%" + str(len(mapping)).encode() + b"\r\n"
    for key, value in mapping.items():
        frame += _bulk(key)
        if isinstance(value, int):
            frame += b":" + str(value).encode() + b"\r\n"
        else:
            frame += _bulk(value)
    return frame


def _push(items):
    frame = b">" + str(len(items)).encode() + b"\r\n"
    for item in items:
        if item is None:
            # RESP3 null: the second element of a full invalidation is null, not an
            # empty array, so the client decodes it as Python ``None``.
            frame += b"_\r\n"
        elif isinstance(item, list):
            frame += _array(item)
        else:
            frame += _bulk(item)
    return frame


def _read_command(buffer):
    line = buffer.readline()
    if not line:
        return None
    argc = int(line[1:])
    args = []
    for _ in range(argc):
        length = int(buffer.readline()[1:])
        args.append(buffer.read(length))
        buffer.read(2)
    return args


class FakeTrackingServer:
    """
    A minimal RESP3 server speaking just the commands the cache handshake and a
    cached ``GET`` need. It hands out a fresh socketpair per connection session and
    keeps serving after the client drops one, so a reconnect is observable as a new
    session with its own command log and its own ``CLIENT TRACKING`` reply.
    """

    def __init__(self, values=None, fail_tracking_on=()):
        self.values = dict(values or {})
        # 1-based session numbers whose CLIENT TRACKING ON must answer with an error.
        self.fail_tracking_on = set(fail_tracking_on)
        self.commands = []
        self.session = 0
        self._lock = threading.Lock()
        self._server_sock = None
        self._all_server_socks = []
        self._all_client_socks = []
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.client_sock = self.open_session()

    def open_session(self):
        client_sock, server_sock = socket.socketpair()
        with self._lock:
            previous = self._server_sock
            self._server_sock = server_sock
            self._all_server_socks.append(server_sock)
            self._all_client_socks.append(client_sock)
            self.session += 1
        if previous is not None:
            try:
                previous.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            previous.close()
        self._wake.set()
        return client_sock

    def push_invalidation(self, keys):
        """Send an invalidation push: a key list, or ``None`` for a full flush."""
        with self._lock:
            self._server_sock.sendall(_push([b"invalidate", keys]))

    def stop(self):
        self._stop = True
        with self._lock:
            current = self._server_sock
        if current is not None:
            try:
                current.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for socks in (self._all_server_socks, self._all_client_socks):
            for opened in socks:
                try:
                    opened.close()
                except OSError:
                    pass
        self._thread.join(timeout=1)

    def _run(self):
        while not self._stop:
            with self._lock:
                server_sock = self._server_sock
                session_no = self.session
            if server_sock is None:
                self._wake.wait(0.1)
                continue
            self._serve_session(server_sock, session_no)

    def _serve_session(self, server_sock, session_no):
        buffer = server_sock.makefile("rb")
        while not self._stop:
            try:
                args = _read_command(buffer)
            except OSError:
                return
            if args is None:
                return
            if session_no != self.session:
                return
            with self._lock:
                self.commands.append(
                    tuple(arg.decode(errors="replace") for arg in args)
                )
                try:
                    self._reply(server_sock, args, session_no)
                except OSError:
                    return

    def _reply(self, server_sock, args, session_no):
        command = args[0].upper()
        if command == b"HELLO":
            server_sock.sendall(
                _map(
                    {
                        b"server": b"redis",
                        b"version": b"7.4.0",
                        b"proto": 3,
                    }
                )
            )
        elif command == b"CLIENT" and args[1].upper() == b"TRACKING":
            if session_no in self.fail_tracking_on:
                server_sock.sendall(b"-ERR tracking refused on this session\r\n")
            else:
                server_sock.sendall(b"+OK\r\n")
        elif command == b"CLIENT":
            server_sock.sendall(b"+OK\r\n")
        elif command == b"PING":
            server_sock.sendall(_bulk(b"PONG"))
        elif command == b"GET":
            server_sock.sendall(_bulk(self.values.get(args[1])))
        else:
            server_sock.sendall(b"+OK\r\n")


def _build_proxy(server, cache=None, **cache_kwargs):
    """
    Wire a ``CacheProxyConnection`` to the fake server by overriding ``_connect``.

    The first connect reuses the server's initial socketpair; every later ``_connect``
    opens a new session, which is exactly the socket shape a silent reconnect has.
    ``network`` is a mutable switch tests flip to make reconnects fail at the socket
    layer (``down``) and then recover (``up``).
    """
    connection = Connection(protocol=3, retry=Retry(NoBackoff(), 0))
    state = {"first": True}
    network = {"up": True}

    def _connect():
        if state["first"]:
            state["first"] = False
            return server.client_sock
        if not network["up"]:
            raise OSError("network unreachable")
        return server.open_session()

    connection._connect = _connect
    if cache is None:
        cache = DefaultCache(CacheConfig(max_size=128, **cache_kwargs))
    proxy = CacheProxyConnection(connection, cache, threading.RLock())
    return proxy, cache, network


def _client(proxy):
    client = redis.Redis()
    client.connection = proxy
    client._single_connection_client = True
    return client


@pytest.fixture()
def tracking_server():
    server = FakeTrackingServer({b"perm": b"allow"})
    yield server
    server.stop()


@pytest.mark.fixed_client
class TestTrackingConnectionLifecycle:
    """How a tracking session and the entries cached through it are related."""

    def test_a_cached_entry_records_the_connection_that_fetched_it(
        self, tracking_server
    ):
        proxy, cache, _ = _build_proxy(tracking_server)
        proxy.connect()
        try:
            proxy.send_command("GET", "perm", keys=["perm"])
            assert proxy.read_response() == b"allow"

            key = CacheKey(
                command="GET", redis_keys=("perm",), redis_args=("GET", "perm")
            )
            entry = cache.get(key)
            # The reverse link a later hit uses to drain the owning connection's
            # parked invalidations before trusting the reply.
            assert entry.status is CacheEntryStatus.VALID
            assert entry.connection_ref is proxy._conn
        finally:
            proxy.disconnect()

    def test_first_connect_enables_tracking_before_any_read(self, tracking_server):
        proxy, _, _ = _build_proxy(tracking_server)
        try:
            proxy.connect()
            # The tracking handshake is part of the connect flow and so completes before
            # the connection can run any data command.
            assert tracking_server.commands[-1] == ("CLIENT", "TRACKING", "ON")
            assert all(command[0] != "GET" for command in tracking_server.commands)
        finally:
            proxy.disconnect()

    def test_a_normal_get_hit_never_reaches_the_server(self, tracking_server):
        proxy, cache, _ = _build_proxy(tracking_server)
        client = _client(proxy)
        try:
            assert client.get("perm") == b"allow"

            # The second read is resolved locally: send_command writes nothing and parks
            # the cached reply for read_response to hand back.
            proxy.send_command("GET", "perm", keys=["perm"])
            assert proxy._pending_cache_hit is not _NO_PENDING_HIT
            assert proxy.read_response() == b"allow"

            assert cache.size == 1
            assert [c for c in tracking_server.commands if c[0] == "GET"] == [
                ("GET", "perm")
            ]
        finally:
            proxy.disconnect()


@pytest.mark.fixed_client
class TestReconnectEviction:
    """
    Entries cached through a tracking session whose channel is gone must be evicted on
    the reconnect, so a changed permission cannot keep being served as a cached result.
    """

    def test_silent_reconnect_flushes_and_refetches_the_new_value(
        self, tracking_server
    ):
        proxy, cache, _ = _build_proxy(tracking_server)
        client = _client(proxy)
        try:
            assert client.get("perm") == b"allow"
            assert client.get("perm") == b"allow"
            assert cache.size == 1

            # The value changes on the server, then the wrapped socket is killed the
            # way the silent send-path reconnect does - CacheProxyConnection.disconnect
            # never runs, so only the connect callback can evict the stale entry.
            tracking_server.values[b"perm"] = b"deny"
            proxy._conn.disconnect()
            assert cache.size == 1

            assert client.get("perm") == b"deny"
            assert cache.size == 1

            tracking_handshakes = [
                command
                for command in tracking_server.commands
                if command[:2] == ("CLIENT", "TRACKING")
            ]
            assert tracking_handshakes == [
                ("CLIENT", "TRACKING", "ON"),
                ("CLIENT", "TRACKING", "ON"),
            ]
        finally:
            proxy.disconnect()

    def test_explicit_disconnect_evicts_before_the_socket_closes(self, tracking_server):
        proxy, cache, _ = _build_proxy(tracking_server)
        client = _client(proxy)
        try:
            client.get("perm")
            assert cache.size == 1

            proxy.disconnect()
            assert cache.size == 0
            assert proxy.is_connected is False
        finally:
            proxy.disconnect()

    def test_re_enabling_tracking_failure_still_evicts_the_old_session(
        self,
    ):
        server = FakeTrackingServer({b"perm": b"allow"}, fail_tracking_on={2})
        proxy, cache, _ = _build_proxy(server)
        client = _client(proxy)
        try:
            assert client.get("perm") == b"allow"
            assert cache.size == 1
            server.values[b"perm"] = b"deny"

            proxy._conn.disconnect()
            # The flush runs before the CLIENT TRACKING handshake is attempted, so the
            # handshake failing leaves nothing from the dead session behind.
            with pytest.raises(ResponseError, match="tracking refused"):
                client.get("perm")
            assert cache.size == 0

            # Once tracking can be enabled again, the read goes to the new session and
            # returns the current value rather than anything cached earlier.
            server.fail_tracking_on.clear()
            assert client.get("perm") == b"deny"
            assert cache.size == 1
        finally:
            proxy.disconnect()
            server.stop()

    def test_socket_level_reconnect_failure_evicts_on_recovery(
        self,
    ):
        server = FakeTrackingServer({b"perm": b"allow"})
        proxy, cache, network = _build_proxy(server)
        client = _client(proxy)
        try:
            assert client.get("perm") == b"allow"
            assert cache.size == 1
            server.values[b"perm"] = b"deny"

            # The network drops before a new socket can be opened. The client-level
            # retry's failure handler calls the proxy's disconnect(), which flushes the
            # entries the unreachable tracking session can no longer invalidate.
            network["up"] = False
            proxy._conn.disconnect()
            with pytest.raises(RedisConnectionError):
                client.get("perm")
            assert cache.size == 0

            # Recovery opens a fresh tracking session and re-fetches the current value.
            network["up"] = True
            assert client.get("perm") == b"deny"
            assert cache.size == 1
        finally:
            proxy.disconnect()
            server.stop()

    def test_an_in_flight_placeholder_survives_the_reconnect_flush(
        self, tracking_server
    ):
        proxy, cache, _ = _build_proxy(tracking_server)
        try:
            proxy.connect()
            key = CacheKey(
                command="GET", redis_keys=("perm",), redis_args=("GET", "perm")
            )
            cache.set(
                CacheEntry(
                    cache_key=key,
                    cache_value=proxy.DUMMY_CACHE_VALUE,
                    status=CacheEntryStatus.IN_PROGRESS,
                    connection_ref=proxy._conn,
                )
            )
            proxy._current_command_cache_key = key
            assert cache.size == 1

            tracking_server.values[b"perm"] = b"deny"
            proxy._conn.disconnect()
            proxy.connect()

            entry = cache.get(key)
            assert entry is not None
            assert entry.status is CacheEntryStatus.IN_PROGRESS
            assert entry.connection_ref is proxy._conn

            # Completing the read against the new session promotes the fresh reply.
            proxy.send_command("GET", "perm", keys=["perm"])
            assert proxy.read_response() == b"deny"
            promoted = cache.get(key)
            assert promoted.status is CacheEntryStatus.VALID
            assert promoted.cache_value == b"deny"
        finally:
            proxy.disconnect()


@pytest.mark.fixed_client
class TestInvalidationCoverage:
    """
    While a tracking session stays live, the server's push channel is the valid
    invalidation cover: entries named by a key push or covered by a full flush are
    evicted through it, and every other entry may stay.
    """

    def test_a_keyed_invalidation_evicts_only_the_named_entry(self, tracking_server):
        proxy, cache, _ = _build_proxy(tracking_server)
        client = _client(proxy)
        try:
            tracking_server.values[b"other"] = b"x"
            client.get("perm")
            client.get("other")
            assert cache.size == 2

            tracking_server.push_invalidation([b"perm"])
            assert proxy.can_read()
            proxy._process_pending_invalidations()

            assert cache.size == 1
            # The untouched entry still has a live channel and stays cached; a read of
            # the invalidated key goes back to the wire instead of serving a stale hit.
            get_commands = [
                command
                for command in tracking_server.commands
                if command == ("GET", "perm")
            ]
            assert client.get("perm") == b"allow"
            assert (
                len(
                    [
                        command
                        for command in tracking_server.commands
                        if command == ("GET", "perm")
                    ]
                )
                == len(get_commands) + 1
            )
        finally:
            proxy.disconnect()

    def test_a_full_invalidation_flushes_every_entry(self, tracking_server):
        proxy, cache, _ = _build_proxy(tracking_server)
        client = _client(proxy)
        try:
            tracking_server.values[b"other"] = b"x"
            client.get("perm")
            client.get("other")
            assert cache.size == 2

            # A null key list is the server's "everything may have changed" notice.
            tracking_server.push_invalidation(None)
            assert proxy.can_read()
            proxy._process_pending_invalidations()

            assert cache.size == 0
        finally:
            proxy.disconnect()

    def test_an_idle_connection_without_pushes_keeps_the_entry(self, tracking_server):
        proxy, cache, _ = _build_proxy(tracking_server)
        client = _client(proxy)
        try:
            client.get("perm")
            assert cache.size == 1

            # The normal drain exit (nothing buffered) must not evict anything: the
            # tracking channel is still the entry's valid cover.
            proxy._process_pending_invalidations()
            assert cache.size == 1
        finally:
            proxy.disconnect()


def _predicate(prefix):
    def predicate(command, keys):
        return redis.utils.str_if_bytes(keys[0]).startswith(prefix)

    return predicate


@pytest.mark.fixed_client
class TestTrackingModePairing:
    """The CLIENT CACHING pairing each mode requires, observed on the wire."""

    def test_plain_mode_tracks_and_caches_without_a_caching_pair(self, tracking_server):
        proxy, cache, _ = _build_proxy(tracking_server)
        client = _client(proxy)
        try:
            client.get("perm")
            data_commands = [
                command
                for command in tracking_server.commands
                if command[0] in ("CLIENT", "GET")
                and not (command[0] == "CLIENT" and command[1] == "SETINFO")
            ]
            assert data_commands == [
                ("CLIENT", "TRACKING", "ON"),
                ("GET", "perm"),
            ]
            assert cache.size == 1
        finally:
            proxy.disconnect()

    def test_optin_pairs_caching_yes_only_for_a_selected_read(self):
        server = FakeTrackingServer({b"user:1": b"alice", b"session:1": b"token"})
        proxy, cache, _ = _build_proxy(
            server,
            tracking_mode=TrackingMode.OPTIN,
            cache_predicate=_predicate("user:"),
        )
        client = _client(proxy)
        try:
            # Selected read: CLIENT CACHING YES immediately before the GET, one write.
            assert client.get("user:1") == b"alice"
            # Not selected: bare GET, never stored, so a repeat reaches the server again.
            assert client.get("session:1") == b"token"
            assert client.get("session:1") == b"token"

            assert ("CLIENT", "CACHING", "YES") in server.commands
            assert ("CLIENT", "TRACKING", "ON", "OPTIN") in server.commands
            assert cache.size == 1
            session_gets = [
                command
                for command in server.commands
                if command == ("GET", "session:1")
            ]
            assert len(session_gets) == 2
        finally:
            proxy.disconnect()
            server.stop()

    def test_optout_pairs_caching_no_for_an_excluded_trackable_read(self):
        server = FakeTrackingServer({b"user:1": b"alice", b"counter:1": b"7"})

        def predicate(command, keys):
            return not redis.utils.str_if_bytes(keys[0]).startswith("counter:")

        proxy, cache, _ = _build_proxy(
            server,
            tracking_mode=TrackingMode.OPTOUT,
            cache_predicate=predicate,
        )
        client = _client(proxy)
        try:
            assert client.get("user:1") == b"alice"
            assert client.get("counter:1") == b"7"

            assert ("CLIENT", "TRACKING", "ON", "OPTOUT") in server.commands
            assert ("CLIENT", "CACHING", "NO") in server.commands
            # Only the stored, still-tracked read remains in the cache.
            assert cache.size == 1
        finally:
            proxy.disconnect()
            server.stop()

    def test_a_reconnect_resends_the_mode_in_the_handshake(self):
        server = FakeTrackingServer({b"user:1": b"alice"})
        proxy, _, _ = _build_proxy(
            server,
            tracking_mode=TrackingMode.OPTIN,
            cache_predicate=_predicate("user:"),
        )
        client = _client(proxy)
        try:
            proxy.connect()
            proxy._conn.disconnect()
            assert client.get("user:1") == b"alice"

            handshakes = [
                command
                for command in server.commands
                if command[:2] == ("CLIENT", "TRACKING")
            ]
            assert handshakes == [
                ("CLIENT", "TRACKING", "ON", "OPTIN"),
                ("CLIENT", "TRACKING", "ON", "OPTIN"),
            ]
        finally:
            proxy.disconnect()
            server.stop()


@pytest.mark.fixed_client
class TestRefusedTrackingCommands:
    """Application-sent commands that would alter tracking state are refused."""

    @pytest.mark.parametrize(
        "command",
        [
            ("CLIENT", "CACHING", "YES"),
            ("CLIENT", "TRACKING", "OFF"),
            ("RESET",),
        ],
    )
    def test_a_tracking_state_command_is_refused(self, tracking_server, command):
        proxy, _, _ = _build_proxy(tracking_server)
        client = _client(proxy)
        try:
            with pytest.raises(RedisError):
                client.execute_command(*command)
        finally:
            proxy.disconnect()

    def test_a_refused_command_writes_nothing(self, tracking_server):
        proxy, _, _ = _build_proxy(tracking_server)
        client = _client(proxy)
        try:
            before = len(tracking_server.commands)
            with pytest.raises(RedisError):
                client.execute_command("CLIENT", "TRACKING", "OFF")
            # Refused before anything reaches the fake server, and the connection stays
            # usable for a normal read.
            assert len(tracking_server.commands) == before
            assert client.get("perm") == b"allow"
        finally:
            proxy.disconnect()
