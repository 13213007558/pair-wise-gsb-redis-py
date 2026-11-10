"""RESP3 push interleaving for the asynchronous pure-Python parser.

Async counterpart of ``tests/test_parsers/test_resp3_push.py``. A RESP3
connection multiplexes command replies and server-initiated push frames
(``>``) on one stream; these tests feed canned bytes through an in-memory
asyncio-style stream to prove that pushes reach their handlers and never take
the place of a queued command's reply.

Only the pure-Python ``_AsyncRESP3Parser`` is exercised here; the async
hiredis parser has its own framing and separate coverage.
"""

import asyncio
from unittest import mock

import pytest
from redis._parsers import _AsyncRESP3Parser
from redis.asyncio import Redis
from redis.asyncio.connection import Connection, ConnectionPool

SNOWMAN = b"\xe2\x98\x83"

PUBSUB_PUSH = b">3\r\n$7\r\nmessage\r\n$2\r\nch\r\n$5\r\nhello\r\n"
INVALIDATION_PUSH = b">2\r\n$10\r\ninvalidate\r\n*1\r\n$3\r\nfoo\r\n"
UNKNOWN_PUSH = b">2\r\n$11\r\ncustom-type\r\n$7\r\npayload\r\n"
MAINTENANCE_PUSH = b">3\r\n$9\r\nMIGRATING\r\n:1\r\n:10\r\n"


def bulk(payload: bytes) -> bytes:
    return b"$%d\r\n%s\r\n" % (len(payload), payload)


class MockStream:
    """Minimal asyncio stream reader backed by a fixed byte string.

    Reads are capped at five bytes per call, like the existing MockStream in
    ``tests/test_asyncio/mocks.py``, so multi-chunk framing is exercised.
    """

    def __init__(self, data):
        self.data = data
        self.pos = 0

    async def read(self, want):
        want = 5
        result = self.data[self.pos : self.pos + want]
        self.pos += len(result)
        return result

    async def readline(self):
        find = self.data.find(b"\n", self.pos)
        if find >= 0:
            result = self.data[self.pos : find + 1]
        else:
            result = self.data[self.pos :]
        self.pos += len(result)
        return result

    async def readexactly(self, length):
        result = self.data[self.pos : self.pos + length]
        if len(result) < length:
            raise asyncio.IncompleteReadError(result, None)
        self.pos += len(result)
        return result

    def at_eof(self):
        return False

    def exception(self):
        return None


_connections: list[Connection] = []


def make_connection(data, **kwargs):
    """A RESP3 async Connection whose reader serves `data`, no real server."""
    kwargs.setdefault("protocol", 3)
    kwargs.setdefault("parser_class", _AsyncRESP3Parser)
    conn = Connection(**kwargs)
    conn._reader = MockStream(data)
    conn._writer = mock.Mock()
    conn._writer.close = mock.Mock()
    conn._writer.wait_closed = mock.AsyncMock()
    conn._parser.on_connect(conn)
    _connections.append(conn)
    return conn


@pytest.fixture(autouse=True)
async def _close_connections():
    """Disconnect every Connection a test built so the garbage collector never
    reports an unclosed connection."""
    yield
    while _connections:
        await _connections.pop().disconnect()


class CaptureHandler:
    """Async push handler that records every push it is handed."""

    def __init__(self):
        self.received = []

    async def __call__(self, response):
        self.received.append(response)
        return response


class PushHandlerError(Exception):
    """Raised by deliberately failing push handlers in the tests below."""


async def failing_handler(response):
    raise PushHandlerError("handler bug")


@pytest.mark.fixed_client
class TestAsyncPushIsNotACommandReply:
    async def test_normal_push_normal_routes_push_to_handler(self):
        handler = CaptureHandler()
        conn = make_connection(
            bulk(b"sentinel-1") + PUBSUB_PUSH + bulk(b"sentinel-2"),
            decode_responses=True,
        )
        conn._parser.set_pubsub_push_handler(handler)

        assert await conn.read_response() == "sentinel-1"
        assert await conn.read_response() == "sentinel-2"

        assert handler.received == [["message", "ch", "hello"]]
        assert conn._parser._buffer == b""
        assert conn._parser._chunks == []

    async def test_consecutive_pushes_before_a_reply(self):
        handler = CaptureHandler()
        conn = make_connection(PUBSUB_PUSH + UNKNOWN_PUSH + bulk(b"after-pushes"))
        conn._parser.set_pubsub_push_handler(handler)

        assert await conn.read_response() == b"after-pushes"
        assert handler.received == [
            [b"message", b"ch", b"hello"],
            [b"custom-type", b"payload"],
        ]
        assert conn._parser._buffer == b""
        assert conn._parser._chunks == []

    async def test_invalidation_push_uses_invalidation_handler(self):
        received = []
        conn = make_connection(INVALIDATION_PUSH + bulk(b"reply"))
        conn._parser.set_invalidation_push_handler(
            lambda response: received.append(response)
        )

        assert await conn.read_response() == b"reply"
        assert received == [[b"invalidate", [b"foo"]]]
        assert conn._parser._buffer == b""

    async def test_maintenance_push_uses_maintenance_handler(self):
        from redis.maint_notifications import NodeMigratingNotification

        received = []
        conn = make_connection(MAINTENANCE_PUSH + bulk(b"reply"))

        async def maintenance_handler(notification):
            received.append(notification)

        conn._parser.set_maintenance_push_handler(maintenance_handler)

        assert await conn.read_response() == b"reply"
        assert len(received) == 1
        assert isinstance(received[0], NodeMigratingNotification)
        assert received[0].id == 1
        assert conn._parser._buffer == b""

    async def test_unknown_push_type_never_returned_as_reply(self):
        conn = make_connection(UNKNOWN_PUSH + bulk(b"command-reply"))

        assert await conn.read_response() == b"command-reply"
        assert conn._parser._buffer == b""

    async def test_push_request_returns_the_push_itself(self):
        conn = make_connection(PUBSUB_PUSH + bulk(b"next-command"))

        assert await conn.read_response(push_request=True) == [
            b"message",
            b"ch",
            b"hello",
        ]
        assert await conn.read_response() == b"next-command"


@pytest.mark.fixed_client
class TestAsyncDecodingFlagsSurvivePushInterleaving:
    async def test_decode_responses_applies_after_push(self):
        handler = CaptureHandler()
        conn = make_connection(
            bulk(SNOWMAN) + PUBSUB_PUSH + bulk(SNOWMAN),
            decode_responses=True,
        )
        conn._parser.set_pubsub_push_handler(handler)

        assert await conn.read_response() == "☃"
        assert await conn.read_response() == "☃"
        assert handler.received == [["message", "ch", "hello"]]

    async def test_single_disable_decoding_survives_interleaved_push(self):
        conn = make_connection(
            bulk(SNOWMAN) + PUBSUB_PUSH + bulk(SNOWMAN),
            decode_responses=True,
        )

        assert await conn.read_response(disable_decoding=True) == SNOWMAN
        assert await conn.read_response(disable_decoding=True) == SNOWMAN

    async def test_decoding_flag_can_change_per_call_across_push(self):
        conn = make_connection(
            bulk(SNOWMAN) + PUBSUB_PUSH + bulk(SNOWMAN),
            decode_responses=True,
        )

        assert await conn.read_response() == "☃"
        assert await conn.read_response(disable_decoding=True) == SNOWMAN


@pytest.mark.fixed_client
class TestAsyncPushHandlerFailure:
    async def test_pubsub_handler_error_without_disconnect_leaves_frame_queued(
        self,
    ):
        # A pub/sub handler failure propagates to the caller. With
        # disconnect_on_error=False the socket stays connected, but the parser
        # has already discarded its assembled buffer: the push bytes remain in
        # the underlying stream (or the parser chunk list), so re-reading
        # re-parses the same push and fails identically rather than returning a
        # push as a command result.
        conn = make_connection(PUBSUB_PUSH + bulk(b"second"))
        conn._parser.pubsub_push_handler_func = failing_handler

        with pytest.raises(PushHandlerError):
            await conn.read_response(disconnect_on_error=False)
        assert conn.is_connected is True

        # The parser cleared its assembled buffer but kept the consumed push
        # chunks; read_response re-augments the buffer from them at _pos = 0.
        # The following command reply was never touched and stays on the stream.
        assert conn._parser._buffer == b""
        assert b"".join(conn._parser._chunks) == PUBSUB_PUSH
        assert conn._reader.data[conn._reader.pos :] == bulk(b"second")

        with pytest.raises(PushHandlerError):
            await conn.read_response(disconnect_on_error=False)

    async def test_pubsub_handler_error_disconnects_by_default(self):
        conn = make_connection(PUBSUB_PUSH + bulk(b"second"))
        conn._parser.pubsub_push_handler_func = failing_handler

        with pytest.raises(PushHandlerError):
            await conn.read_response()
        assert conn.is_connected is False

    async def test_invalidation_handler_error_is_swallowed_and_reused(self):
        # handle_push_response catches failures from the invalidation and
        # maintenance handlers itself: the connection stays up and the next
        # command reply is returned normally.
        conn = make_connection(INVALIDATION_PUSH + bulk(b"second"))
        conn._parser.set_invalidation_push_handler(failing_handler)

        assert await conn.read_response(disconnect_on_error=False) == b"second"
        assert conn.is_connected is True
        assert conn._parser._buffer == b""


@pytest.mark.fixed_client
class TestAsyncPushInterleavingThroughConnectionAndPipeline:
    """End-to-end reads through the async Connection/pipeline with a distinct
    sentinel reply per command, so a push cannot silently take a reply's slot.
    """

    def _pipeline_connection(self, stream):
        conn = make_connection(stream, decode_responses=True)
        conn.send_packed_command = mock.AsyncMock()

        pool = mock.Mock(spec=ConnectionPool)
        pool.get_connection = mock.AsyncMock(return_value=conn)
        pool.release = mock.AsyncMock()

        client = Redis(protocol=3)
        client.connection_pool = pool
        return conn, client

    async def test_pipeline_keeps_push_and_command_results_apart(self):
        invalidations = []
        pubsub_handler = CaptureHandler()
        stream = (
            bulk(b"sentinel-1")
            + INVALIDATION_PUSH
            + bulk(b"sentinel-2")
            + PUBSUB_PUSH
            + bulk(b"sentinel-3")
        )
        conn, client = self._pipeline_connection(stream)
        conn._parser.set_invalidation_push_handler(
            lambda response: invalidations.append(response)
        )
        conn._parser.set_pubsub_push_handler(pubsub_handler)

        pipe = client.pipeline(transaction=False)
        pipe.get("a").get("b").get("c")
        assert await pipe.execute() == [
            "sentinel-1",
            "sentinel-2",
            "sentinel-3",
        ]

        assert invalidations == [["invalidate", ["foo"]]]
        assert pubsub_handler.received == [["message", "ch", "hello"]]
        assert conn.is_connected is True
        assert conn._parser._buffer == b""

    async def test_consecutive_pushes_inside_pipeline_dont_shift_results(self):
        pubsub_handler = CaptureHandler()
        stream = bulk(b"sentinel-1") + PUBSUB_PUSH + UNKNOWN_PUSH + bulk(b"sentinel-2")
        conn, client = self._pipeline_connection(stream)
        conn._parser.set_pubsub_push_handler(pubsub_handler)

        pipe = client.pipeline(transaction=False)
        pipe.get("a").get("b")
        assert await pipe.execute() == ["sentinel-1", "sentinel-2"]
        assert pubsub_handler.received == [
            ["message", "ch", "hello"],
            ["custom-type", "payload"],
        ]
        assert conn.is_connected is True
        assert conn._parser._buffer == b""

    async def test_connection_pair_of_commands_with_push_between(self):
        handler = CaptureHandler()
        conn = make_connection(
            bulk(b"sentinel-1") + PUBSUB_PUSH + bulk(b"sentinel-2"),
            decode_responses=True,
        )
        conn._parser.set_pubsub_push_handler(handler)
        conn.send_command = mock.AsyncMock()

        first = await conn.read_response()
        second = await conn.read_response()
        assert (first, second) == ("sentinel-1", "sentinel-2")
        assert handler.received == [["message", "ch", "hello"]]
        assert conn._parser._buffer == b""
