"""Async counterpart of tests/test_reauth_rotation.py.

No Redis server and no real sockets: a recording asyncio-style reader/writer
stands in for the transport and a scripted streaming credential provider drives
rotation. Each assertion is made against a per-connection transcript so the
ordering guarantees (a business reply is never consumed as an AUTH reply, a
connection is only lendable again once its own AUTH succeeded) are explicit.
"""

import asyncio
from unittest import mock

import pytest
import redis.asyncio as redis_async
from redis import AuthenticationError
from redis.asyncio.connection import Connection, ConnectionPool
from redis.auth.token import SimpleToken
from redis.credentials import StreamingCredentialProvider
from redis.event import (
    AfterPooledConnectionsInstantiationEvent,
    AsyncAfterConnectionReleasedEvent,
    ClientType,
    EventException,
    RegisterReAuthForPooledConnections,
)
from redis.exceptions import ConnectionError as RedisConnectionError

pytestmark = pytest.mark.fixed_client


def make_token(oid: str, secret: str) -> SimpleToken:
    return SimpleToken(
        value=secret,
        expires_at_ms=-1,
        received_at_ms=0,
        claims={"oid": oid},
    )


class ScriptedStreamingProvider(StreamingCredentialProvider):
    def __init__(self, initial_oid="oid-1", initial_secret="secret-1"):
        self._on_next = None
        self._on_error = None
        self.initial = (initial_oid, initial_secret)

    def get_credentials(self):
        return self.initial

    async def get_credentials_async(self):
        return self.initial

    def on_next(self, callback):
        self._on_next = callback

    def on_error(self, callback):
        self._on_error = callback

    def is_streaming(self) -> bool:
        return True

    async def rotate(self, token):
        await self._on_next(token)

    async def fail(self, error):
        await self._on_error(error)


class RecordingWriter:
    """Fake asyncio.StreamWriter that records writes and can fail on demand."""

    def __init__(self, events):
        self.events = events
        self.transport = mock.Mock()
        self.transport.is_closing.return_value = False
        self._fail_writelines = None

    def writelines(self, chunks):
        data = b"".join(chunks)
        self.events.append(("send", data))
        if self._fail_writelines is not None:
            error = self._fail_writelines
            self._fail_writelines = None
            raise error

    def write(self, data):
        self.events.append(("send", bytes(data)))

    def fail_next_write(self, error):
        self._fail_writelines = error

    def close(self):
        self.events.append(("writer-close", None))

    async def wait_closed(self):
        self.events.append(("wait-closed", None))

    async def drain(self):
        return None


class RecordingReader:
    """Minimal asyncio.StreamReader substitute serving scripted RESP chunks."""

    def __init__(self, events, chunks):
        self.events = events
        self._buffer = bytearray()
        self._buf = self._buffer
        for chunk in chunks:
            self._buffer.extend(chunk)
        self._eof = False
        self._exc = None

    def at_eof(self):
        return self._eof and not self._buffer

    def exception(self):
        return self._exc

    async def readline(self):
        idx = self._buffer.find(b"\n")
        if idx < 0:
            return b""
        out = bytes(self._buffer[: idx + 1])
        del self._buffer[: idx + 1]
        self.events.append(("recv", out))
        return out

    async def readexactly(self, n):
        if len(self._buffer) < n:
            raise asyncio.IncompleteReadError(bytes(self._buffer), n)
        out = bytes(self._buffer[:n])
        del self._buffer[:n]
        self.events.append(("recv", out))
        return out

    async def read(self, n=-1):
        out = bytes(self._buffer)
        self._buffer.clear()
        return out


def build_connection(name, replies=()):
    events = []
    connection = Connection(protocol=2, health_check_interval=0)
    reader = RecordingReader(events, replies)
    writer = RecordingWriter(events)
    connection._reader = reader
    connection._writer = writer
    connection._parser.on_connect(connection)
    connection._test_name = name
    return connection, events, writer


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


def transcript(events):
    return [(kind, payload) for kind, payload in events if kind in ("send", "recv")]


class TestIdleConnectionRotation:
    async def test_idle_connection_authenticates_before_it_is_lent_again(self):
        conn, events, _ = build_connection("idle", replies=[OK])
        pool = build_pool(conn)

        await pool.re_auth_callback(make_token("oid-2", "secret-2"))

        assert transcript(events) == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]
        assert pool._available_connections == [conn]
        assert conn._re_auth_token is None
        assert conn.is_connected

    async def test_release_event_reauthenticates_idle_connection_with_pending_token(
        self,
    ):
        conn, events, _ = build_connection("released", replies=[OK])
        pool = build_pool(conn)
        pool._available_connections.remove(conn)
        pool._in_use_connections.add(conn)
        conn.set_re_auth_token(make_token("oid-2", "secret-2"))

        await pool.release(conn)

        assert transcript(events) == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]
        assert conn._re_auth_token is None
        assert conn in pool._available_connections


class TestInFlightPipelineRotation:
    async def test_business_reply_is_consumed_before_release_triggers_auth(self):
        conn, events, _ = build_connection("pipeline", replies=[b"+PONG\r\n", OK])
        pool = build_pool(conn)
        pool._available_connections.remove(conn)
        pool._in_use_connections.add(conn)
        token = make_token("oid-2", "secret-2")

        await pool.re_auth_callback(token)
        assert transcript(events) == []
        assert conn._re_auth_token is token

        assert await conn.read_response() == b"PONG"
        await pool.release(conn)

        assert transcript(events) == [
            ("recv", b"+PONG\r\n"),
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]
        assert conn._re_auth_token is None

    async def test_re_auth_does_not_interleave_with_a_command_already_sent(self):
        conn, events, _ = build_connection("busy", replies=[b"+PONG\r\n", OK])
        pool = build_pool(conn)
        pool._available_connections.remove(conn)
        pool._in_use_connections.add(conn)

        await conn.send_command("PING")
        await pool.re_auth_callback(make_token("oid-2", "secret-2"))

        # No AUTH bytes may follow PING before its reply is consumed.
        assert transcript(events) == [("send", b"*1\r\n$4\r\nPING\r\n")]
        assert await conn.read_response() == b"PONG"
        await pool.release(conn)
        assert transcript(events)[-2:] == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]


class TestConsecutiveGenerations:
    async def test_two_generations_leave_only_the_newest_identity_pending(self):
        idle, idle_events, _ = build_connection("idle", replies=[OK, OK])
        busy, busy_events, _ = build_connection("busy", replies=[b"+PONG\r\n", OK])
        pool = build_pool(idle, busy)
        pool._available_connections.remove(busy)
        pool._in_use_connections.add(busy)

        first = make_token("oid-2", "secret-2")
        second = make_token("oid-3", "secret-3")
        await pool.re_auth_callback(first)
        await pool.re_auth_callback(second)

        sends = [payload for kind, payload in transcript(idle_events) if kind == "send"]
        assert sends == [
            auth_frame("oid-2", "secret-2"),
            auth_frame("oid-3", "secret-3"),
        ]
        assert busy._re_auth_token is second

        assert await busy.read_response() == b"PONG"
        await pool.release(busy)
        assert [p for k, p in transcript(busy_events) if k == "send"] == [
            auth_frame("oid-3", "secret-3")
        ]


class TestRejectedAuth:
    async def test_authentication_error_evicts_the_connection_from_the_pool(self):
        failing, failing_events, _ = build_connection("failing", replies=[WRONGPASS])
        other, other_events, _ = build_connection("other", replies=[OK])
        in_use, _, _ = build_connection("in_use")
        pool = build_pool(failing, other)
        pool._in_use_connections.add(in_use)

        with pytest.raises(AuthenticationError, match="invalid username-password"):
            await pool.re_auth_callback(make_token("oid-2", "secret-2"))

        assert failing not in pool._available_connections
        assert not failing.is_connected
        assert other in pool._available_connections
        assert transcript(other_events) == []
        assert in_use._re_auth_token is None

    async def test_connection_failure_evicts_the_connection_from_the_pool(self):
        failing, failing_events, failing_writer = build_connection("failing")
        failing_writer.fail_next_write(OSError(32, "Broken pipe"))
        other, other_events, _ = build_connection("other", replies=[OK])
        pool = build_pool(failing, other)

        with pytest.raises(RedisConnectionError, match="Broken pipe"):
            await pool.re_auth_callback(make_token("oid-2", "secret-2"))

        assert failing not in pool._available_connections
        assert not failing.is_connected
        assert transcript(other_events) == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]
        assert other in pool._available_connections

    async def test_release_time_re_auth_failure_evicts_the_connection(self):
        # Rotation tags a busy connection; its socket dies before release, so the
        # release listener's AUTH cannot succeed and the connection is dropped.
        conn, events, _ = build_connection("dying")
        pool = build_pool(conn)
        pool._available_connections.remove(conn)
        pool._in_use_connections.add(conn)
        conn.set_re_auth_token(make_token("oid-2", "secret-2"))
        conn._reader = None
        conn._writer = None
        conn._parser.on_disconnect()

        with pytest.raises((RedisConnectionError, OSError)):
            await pool.release(conn)

        assert conn not in pool._available_connections
        assert conn._re_auth_token is None


class TestDisconnectDuringRotation:
    async def test_checkout_cannot_pass_a_connection_while_idle_re_auth_runs(self):
        # The pool lock serializes idle re-auth against checkout, so a borrower
        # either gets the connection before AUTH starts or after it finished.
        conn, events, _ = build_connection("locked", replies=[OK])
        pool = build_pool(conn)
        token = make_token("oid-2", "secret-2")

        auth_started = asyncio.Event()
        finish_auth = asyncio.Event()

        real_read = conn.read_response

        async def gated_read(*args, **kwargs):
            auth_started.set()
            await finish_auth.wait()
            return await real_read(*args, **kwargs)

        conn.read_response = gated_read
        refresh = asyncio.create_task(pool.re_auth_callback(token))
        await auth_started.wait()

        checkout = asyncio.create_task(pool.get_connection())
        await asyncio.sleep(0)
        assert not checkout.done()
        assert pool._available_connections == [conn]

        finish_auth.set()
        await asyncio.gather(refresh, checkout)
        assert checkout.result() is conn
        assert transcript(events) == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]
        await pool.release(conn)


class TestStreamingProviderWiring:
    async def test_pooled_rotation_is_registered_only_for_streaming_providers(self):
        provider = ScriptedStreamingProvider()
        pool = redis_async.Redis(credential_provider=provider).connection_pool
        conn, events, _ = build_connection("wired", replies=[OK])
        pool._available_connections.append(conn)

        await provider.rotate(make_token("oid-2", "secret-2"))

        assert transcript(events) == [
            ("send", auth_frame("oid-2", "secret-2")),
            ("recv", OK),
        ]
        await pool.disconnect()

    async def test_provider_error_propagates_as_an_event_exception(self):
        provider = ScriptedStreamingProvider()
        pool = ConnectionPool(connection_class=Connection, credential_provider=provider)
        listener = RegisterReAuthForPooledConnections()
        listener.listen(
            AfterPooledConnectionsInstantiationEvent([pool], ClientType.ASYNC, provider)
        )
        with pytest.raises(EventException):
            await provider.fail(RuntimeError("token endpoint down"))
        await pool.disconnect()

    async def test_non_streaming_provider_has_no_background_rotation_path(self):
        # Scope guard: a plain CredentialProvider is only consulted while
        # connecting, so the re-auth listener must not register on_next for it.
        class PlainProvider:
            def get_credentials(self):
                return ("secret-1",)

            async def get_credentials_async(self):
                return ("secret-1",)

        provider = PlainProvider()
        pool = ConnectionPool(connection_class=Connection, credential_provider=provider)
        listener = RegisterReAuthForPooledConnections()
        listener.listen(
            AfterPooledConnectionsInstantiationEvent([pool], ClientType.ASYNC, provider)
        )
        assert listener._event is None
        await pool.disconnect()

    async def test_release_event_class_is_async_for_the_async_stack(self):
        # The async pool must dispatch the async event; a sync listener would
        # attempt a non-awaited coroutine and silently swallow re-auth.
        seen = []

        class Recorder:
            async def listen(self, event):
                assert isinstance(event, AsyncAfterConnectionReleasedEvent)
                seen.append(event.connection)

        pool = ConnectionPool(connection_class=Connection)
        pool._event_dispatcher.register_listeners(
            {AsyncAfterConnectionReleasedEvent: [Recorder()]}
        )
        conn, _, _ = build_connection("recorded")
        pool._in_use_connections.add(conn)
        await pool.release(conn)
        assert seen == [conn]
        await pool.disconnect()
