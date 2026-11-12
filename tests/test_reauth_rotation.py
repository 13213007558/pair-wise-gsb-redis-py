"""Controlled-event tests for credential rotation and pooled re-authentication.

These tests never open a socket: a recording fake transport stands in for the
wire and a scripted token provider drives the rotation. Every assertion is made
against the per-connection transcript of send/receive/close events, so the
ordering guarantees (business replies are never mistaken for AUTH replies, a
connection is only lendable again once its own AUTH succeeded) are checked
directly rather than through a live Redis server.
"""

import pytest
from redis import AuthenticationError
from redis.auth.token import SimpleToken
from redis.connection import Connection, ConnectionPool
from redis.credentials import StreamingCredentialProvider
from redis.exceptions import ConnectionError as RedisConnectionError

pytestmark = pytest.mark.fixed_client


def make_token(oid: str, secret: str) -> SimpleToken:
    # expires_at == -1 marks a non-expiring token, which is all these unit
    # tests need: the pool only reads oid/value.
    return SimpleToken(
        value=secret,
        expires_at_ms=-1,
        received_at_ms=0,
        claims={"oid": oid},
    )


class ScriptedStreamingProvider(StreamingCredentialProvider):
    """Streaming provider whose rotation is driven explicitly by the test."""

    def __init__(self, initial_oid="oid-1", initial_secret="secret-1"):
        self._on_next = None
        self._on_error = None
        self.generation = 0
        self.initial = (initial_oid, initial_secret)

    def get_credentials(self):
        return self.initial

    def on_next(self, callback):
        self._on_next = callback

    def on_error(self, callback):
        self._on_error = callback

    def is_streaming(self) -> bool:
        return True

    def rotate(self, token):
        """Deliver a new credential generation, as a token manager would."""
        self.generation += 1
        self._on_next(token)

    def fail(self, error):
        self._on_error(error)


class RecordingSocket:
    """Fake socket that records the wire transcript and serves scripted replies.

    ``replies`` is consumed one chunk per ``recv``. ``send_error`` makes
    ``sendall`` raise the given OSError, modelling a socket that died while a
    command was in flight.
    """

    def __init__(self, name, replies=(), send_error=None):
        self.name = name
        self.replies = list(replies)
        self.send_error = send_error
        self.events = []
        self._closed = False

    def sendall(self, data):
        self.events.append(("send", bytes(data)))
        if self.send_error is not None:
            error = self.send_error
            self.send_error = None
            raise error

    def recv(self, bufsize):
        if self._closed:
            return b""
        if not self.replies:
            return b""
        chunk = self.replies.pop(0)
        self.events.append(("recv", chunk))
        return chunk

    def settimeout(self, timeout):
        self.events.append(("settimeout", timeout))

    def shutdown(self, how):
        self.events.append(("shutdown", how))

    def close(self):
        self._closed = True
        self.events.append(("close", None))


def build_connection(name, replies=(), send_error=None):
    """A connected-looking Connection whose socket is a RecordingSocket."""
    connection = Connection(protocol=2, health_check_interval=0)
    sock = RecordingSocket(name, replies, send_error)
    connection._sock = sock
    connection._parser.on_connect(connection)
    return connection, sock


def build_pool(*connections):
    pool = ConnectionPool(connection_class=Connection)
    pool._available_connections.extend(connections)
    return pool


def auth_frame(oid, secret):
    return (
        f"*3\r\n$4\r\nAUTH\r\n${len(oid)}\r\n{oid}\r\n"
        f"${len(secret)}\r\n{secret}\r\n"
    ).encode()


OK = b"+OK\r\n"
WRONGPASS = b"-WRONGPASS invalid username-password pair or user is disabled.\r\n"


def transcript(sock):
    """Collapse the wire transcript to (direction, payload) pairs."""
    return [
        (kind, payload) for kind, payload in sock.events if kind in ("send", "recv")
    ]


class TestIdleConnectionRotation:
    def test_idle_connection_authenticates_before_it_is_lent_again(self):
        conn, sock = build_connection("idle", replies=[OK])
        pool = build_pool(conn)
        token = make_token("oid-2", "secret-2")

        pool.re_auth_callback(token)

        # The only bytes on the wire are AUTH with the new generation.
        assert transcript(sock) == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]
        # The connection remains reusable and carries no pending token.
        assert pool._available_connections == [conn]
        assert conn._re_auth_token is None
        assert conn._sock is sock

    def test_release_event_reauthenticates_idle_connection_with_pending_token(self):
        conn, sock = build_connection("released", replies=[OK])
        pool = build_pool(conn)
        # The connection is momentarily checked out, then a rotation marks it
        # while it is in use; release must apply the token before it is reused.
        pool._available_connections.remove(conn)
        pool._in_use_connections.add(conn)
        token = make_token("oid-2", "secret-2")
        conn.set_re_auth_token(token)

        pool.release(conn)

        assert transcript(sock) == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]
        assert conn._re_auth_token is None
        assert pool._available_connections == [conn]


class TestInFlightPipelineRotation:
    def test_business_reply_is_consumed_before_release_triggers_auth(self):
        # Two queued replies: the pipeline's PONG, then the AUTH OK that will
        # be sent only after release.
        conn, sock = build_connection("pipeline", replies=[b"+PONG\r\n", OK])
        pool = build_pool(conn)
        pool._available_connections.remove(conn)
        pool._in_use_connections.add(conn)
        token = make_token("oid-2", "secret-2")

        # Rotation while the pipeline is still executing: the connection is not
        # touched on the wire, it is merely tagged.
        pool.re_auth_callback(token)
        assert transcript(sock) == []
        assert conn._re_auth_token is token

        # Pipeline drains its own business response first.
        assert conn.read_response() == b"PONG"

        # Returning the connection fires re-auth, which reads its own AUTH OK
        # rather than the already-consumed PONG.
        pool.release(conn)
        assert transcript(sock) == [
            ("recv", b"+PONG\r\n"),
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]
        assert conn._re_auth_token is None

    def test_re_auth_does_not_interleave_with_a_command_already_sent(self):
        # While the connection is in use, rotation must never write AUTH onto a
        # socket that has an outstanding command.
        conn, sock = build_connection("busy", replies=[b"+PONG\r\n", OK])
        pool = build_pool(conn)
        pool._available_connections.remove(conn)
        pool._in_use_connections.add(conn)

        conn.send_command("PING")
        pool.re_auth_callback(make_token("oid-2", "secret-2"))

        # No AUTH bytes follow PING before the business reply is read.
        assert transcript(sock) == [
            ("send", b"*1\r\n$4\r\nPING\r\n"),
        ]
        assert conn.read_response() == b"PONG"
        pool.release(conn)
        assert transcript(sock)[-2:] == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]


class TestConsecutiveGenerations:
    def test_two_generations_leave_only_the_newest_identity_pending(self):
        idle, idle_sock = build_connection("idle", replies=[OK, OK])
        busy, busy_sock = build_connection("busy", replies=[b"+PONG\r\n", OK])
        pool = build_pool(idle, busy)
        pool._available_connections.remove(busy)
        pool._in_use_connections.add(busy)

        first = make_token("oid-2", "secret-2")
        second = make_token("oid-3", "secret-3")
        pool.re_auth_callback(first)
        pool.re_auth_callback(second)

        # Idle connection authenticated twice, ending on the newest generation.
        assert idle_sock.events.count(("close", None)) == 0
        sends = [payload for kind, payload in transcript(idle_sock) if kind == "send"]
        assert sends == [
            auth_frame("oid-2", "secret-2"),
            auth_frame("oid-3", "secret-3"),
        ]
        # In-use connection only retains the newest token: no stale AUTH.
        assert busy._re_auth_token is second

        assert busy.read_response() == b"PONG"
        pool.release(busy)
        assert [p for k, p in transcript(busy_sock) if k == "send"] == [
            auth_frame("oid-3", "secret-3")
        ]


class TestRejectedAuth:
    def test_authentication_error_evicts_the_connection_from_the_pool(self):
        failing, failing_sock = build_connection("failing", replies=[WRONGPASS])
        other, other_sock = build_connection("other", replies=[OK])
        pool = build_pool(failing, other)
        in_use = build_connection("in_use")[0]
        pool._in_use_connections.add(in_use)
        token = make_token("oid-2", "secret-2")

        with pytest.raises(AuthenticationError, match="invalid username-password"):
            pool.re_auth_callback(token)

        # The rejected connection must not stay in the reusable set, and its
        # socket is gone: it cannot be lent out still holding the old identity.
        assert failing not in pool._available_connections
        assert failing not in pool._in_use_connections
        assert failing._sock is None
        assert ("close", None) in failing_sock.events
        # The rotation aborts before touching later connections or in-use ones.
        assert transcript(other_sock) == []
        assert in_use._re_auth_token is None
        # The healthy connection stays lendable under its existing identity.
        assert other in pool._available_connections

    def test_connection_failure_evicts_the_connection_from_the_pool(self):
        broken_pipe = OSError(32, "Broken pipe")
        failing, failing_sock = build_connection("failing", send_error=broken_pipe)
        other, other_sock = build_connection("other", replies=[OK])
        pool = build_pool(failing, other)
        token = make_token("oid-2", "secret-2")

        with pytest.raises(RedisConnectionError, match="Broken pipe"):
            pool.re_auth_callback(token)

        # A transport-level failure is not an AUTH success: the connection is
        # out of the reusable set and disconnected, never silently lent again.
        assert failing not in pool._available_connections
        assert failing._sock is None
        assert transcript(other_sock) == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]
        assert other in pool._available_connections

    def test_eviction_reclaims_pool_capacity(self):
        failing, _ = build_connection("failing", replies=[WRONGPASS])
        pool = build_pool(failing)
        pool._created_connections = 1

        with pytest.raises(AuthenticationError):
            pool.re_auth_callback(make_token("oid-2", "secret-2"))

        assert pool._created_connections == 0


class TestDisconnectDuringRotation:
    def test_connection_disconnected_between_refresh_and_release_reconnects_with_new_auth(
        self,
    ):
        # Rotation tags an in-use connection; before release its socket dies.
        conn, sock = build_connection("dying", replies=[])
        pool = build_pool(conn)
        pool._available_connections.remove(conn)
        pool._in_use_connections.add(conn)
        token = make_token("oid-2", "secret-2")
        conn.set_re_auth_token(token)

        conn.disconnect()
        assert conn._sock is None

        # release() puts the connection back, then the release listener calls
        # re_auth(); with no socket the AUTH attempt fails and must evict it
        # rather than resurrecting a half-authenticated connection.
        with pytest.raises((RedisConnectionError, OSError)):
            pool.release(conn)

        assert conn not in pool._available_connections
        assert conn._re_auth_token is None

    def test_checkout_blocks_until_idle_re_auth_finishes(self):
        # The pool lock serializes idle re-auth with checkout, so a borrower
        # can never observe a connection whose AUTH is half written.
        import threading

        conn, sock = build_connection("locked", replies=[OK])
        pool = build_pool(conn)
        token = make_token("oid-2", "secret-2")

        auth_started = threading.Event()
        finish_auth = threading.Event()
        real_read = conn.read_response

        def gated_read(*args, **kwargs):
            auth_started.set()
            finish_auth.wait(timeout=5)
            return real_read(*args, **kwargs)

        conn.read_response = gated_read
        reauth_thread = threading.Thread(target=pool.re_auth_callback, args=(token,))
        reauth_thread.start()
        assert auth_started.wait(timeout=5)

        # A checkout's locked section cannot run while re-auth holds the lock.
        lock_acquired = threading.Event()

        def try_lock():
            with pool._lock:
                lock_acquired.set()

        checkout_thread = threading.Thread(target=try_lock)
        checkout_thread.start()
        checkout_thread.join(timeout=0.2)
        assert checkout_thread.is_alive()
        assert not lock_acquired.is_set()
        assert pool._available_connections == [conn]

        finish_auth.set()
        checkout_thread.join(timeout=5)
        reauth_thread.join(timeout=5)
        assert lock_acquired.is_set()
        assert transcript(sock) == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]


class TestStreamingProviderWiring:
    def test_pooled_rotation_is_registered_only_for_streaming_providers(self):
        provider = ScriptedStreamingProvider()
        pool = ConnectionPool(connection_class=Connection, credential_provider=provider)
        # Simulate the client construction event that registers the callback.
        from redis.event import AfterPooledConnectionsInstantiationEvent, ClientType

        pool._event_dispatcher.dispatch(
            AfterPooledConnectionsInstantiationEvent([pool], ClientType.SYNC, provider)
        )
        conn, sock = build_connection("wired", replies=[OK])
        pool._available_connections.append(conn)

        provider.rotate(make_token("oid-2", "secret-2"))

        assert transcript(sock) == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]

    def test_provider_error_propagates_as_an_event_exception(self):
        from redis.event import (
            EventException,
            RegisterReAuthForPooledConnections,
        )

        provider = ScriptedStreamingProvider()
        pool = ConnectionPool(connection_class=Connection, credential_provider=provider)
        listener = RegisterReAuthForPooledConnections()
        from redis.event import AfterPooledConnectionsInstantiationEvent, ClientType

        listener.listen(
            AfterPooledConnectionsInstantiationEvent([pool], ClientType.SYNC, provider)
        )
        boom = RuntimeError("token endpoint down")
        with pytest.raises(EventException):
            provider.fail(boom)

    def test_non_streaming_provider_has_no_background_rotation_path(self):
        # Scope guard: a plain CredentialProvider is consulted at connect time
        # only, so the client must not register an on_next re-auth callback for
        # it. The event listener simply ignores it.
        from redis.event import (
            AfterPooledConnectionsInstantiationEvent,
            ClientType,
            RegisterReAuthForPooledConnections,
        )

        class PlainProvider:
            def get_credentials(self):
                return ("secret-1",)

        provider = PlainProvider()
        pool = ConnectionPool(connection_class=Connection, credential_provider=provider)
        listener = RegisterReAuthForPooledConnections()
        listener.listen(
            AfterPooledConnectionsInstantiationEvent([pool], ClientType.SYNC, provider)
        )
        # No token was stored on the listener: there is no callback to fire.
        assert listener._event is None
