"""RESP3 push interleaving for the synchronous pure-Python parser.

A RESP3 connection carries ordinary command replies and server-initiated push
frames (``>``) on the same byte stream. A push can arrive ahead of any reply,
between the replies of a pipeline, or several in a row. These tests drive the
parser and the Connection over canned in-memory byte streams to prove that a
push is delivered to its handler and is never returned as the value of a
queued command.

Only the pure-Python ``_RESP3Parser`` is exercised here; the hiredis parser
has its own framing and is covered separately.
"""

import socket
from unittest import mock

import pytest
from redis import Redis
from redis._parsers import _RESP3Parser
from redis.connection import Connection, ConnectionPool

SNOWMAN = b"\xe2\x98\x83"

# A pub/sub push (``>3`` message/channel/data), a client-side cache
# invalidation (``>2`` invalidate/keys) and a push type this client does not
# know about. The unknown type falls through to the pub/sub handler.
PUBSUB_PUSH = b">3\r\n$7\r\nmessage\r\n$2\r\nch\r\n$5\r\nhello\r\n"
INVALIDATION_PUSH = b">2\r\n$10\r\ninvalidate\r\n*1\r\n$3\r\nfoo\r\n"
UNKNOWN_PUSH = b">2\r\n$11\r\ncustom-type\r\n$7\r\npayload\r\n"
MAINTENANCE_PUSH = b">3\r\n$9\r\nMIGRATING\r\n:1\r\n:10\r\n"


def bulk(payload: bytes) -> bytes:
    return b"$%d\r\n%s\r\n" % (len(payload), payload)


class CannedSocket:
    """Serves a fixed byte stream, then behaves like an open-but-idle socket.

    Reads are capped at five bytes so a single protocol frame spans several
    socket reads, exercising the SocketBuffer refill/rewind paths rather than
    parsing every frame in one recv.
    """

    def __init__(self, data):
        self.data = data
        self.timeout = None

    def recv(self, n):
        if not self.data:
            raise socket.timeout("idle")
        n = min(5, n)
        chunk, self.data = self.data[:n], self.data[n:]
        return chunk

    def recv_into(self, buffer, nbytes=0):
        chunk = self.recv(nbytes or len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)

    def settimeout(self, t):
        self.timeout = t

    def gettimeout(self):
        return self.timeout

    def close(self):
        pass

    def shutdown(self, how):
        pass


def make_connection(data, **kwargs):
    """A RESP3 Connection whose socket serves `data`, without a real server."""
    kwargs.setdefault("protocol", 3)
    kwargs.setdefault("parser_class", _RESP3Parser)
    conn = Connection(**kwargs)
    conn._sock = CannedSocket(data)
    conn._parser.on_connect(conn)
    return conn


class PushHandlerError(Exception):
    """Raised by deliberately failing push handlers in the tests below."""


def failing_handler(response):
    raise PushHandlerError("handler bug")


@pytest.mark.fixed_client
class TestPushIsNotACommandReply:
    def test_normal_push_normal_routes_push_to_handler(self):
        # reply, pub/sub push, reply: each read returns a command reply and the
        # push goes only to the handler.
        received = []
        conn = make_connection(
            bulk(b"sentinel-1") + PUBSUB_PUSH + bulk(b"sentinel-2"),
            decode_responses=True,
        )
        conn._parser.set_pubsub_push_handler(
            lambda response: (received.append(response), response)[1]
        )

        assert conn.read_response() == "sentinel-1"
        assert conn.read_response() == "sentinel-2"

        assert received == [["message", "ch", "hello"]]
        assert conn._parser._buffer.unread_bytes() == 0

    def test_consecutive_pushes_before_a_reply(self):
        # Two pushes back to back, then the actual reply. A single
        # read_response must drain both pushes and still land on the reply.
        received = []
        conn = make_connection(PUBSUB_PUSH + UNKNOWN_PUSH + bulk(b"after-pushes"))
        conn._parser.set_pubsub_push_handler(
            lambda response: (received.append(response), response)[1]
        )

        assert conn.read_response() == b"after-pushes"
        assert received == [
            [b"message", b"ch", b"hello"],
            [b"custom-type", b"payload"],
        ]
        assert conn._parser._buffer.unread_bytes() == 0

    def test_invalidation_push_uses_invalidation_handler(self):
        received = []
        conn = make_connection(INVALIDATION_PUSH + bulk(b"reply"))
        conn._parser.set_invalidation_push_handler(
            lambda response: received.append(response)
        )

        assert conn.read_response() == b"reply"
        assert received == [[b"invalidate", [b"foo"]]]
        assert conn._parser._buffer.unread_bytes() == 0

    def test_maintenance_push_uses_maintenance_handler(self):
        from redis.maint_notifications import NodeMigratingNotification

        received = []
        conn = make_connection(MAINTENANCE_PUSH + bulk(b"reply"))
        conn._parser.set_maintenance_push_handler(
            lambda notification: received.append(notification)
        )

        assert conn.read_response() == b"reply"
        assert len(received) == 1
        assert isinstance(received[0], NodeMigratingNotification)
        assert received[0].id == 1
        assert conn._parser._buffer.unread_bytes() == 0

    def test_unknown_push_type_never_returned_as_reply(self):
        # With the default handler an unknown push is merely logged/returned
        # inside the parser; the caller still gets the following command reply.
        conn = make_connection(UNKNOWN_PUSH + bulk(b"command-reply"))

        assert conn.read_response() == b"command-reply"
        assert conn._parser._buffer.unread_bytes() == 0

    def test_push_request_returns_the_push_itself(self):
        # Pub/sub reads ask for the push explicitly; then the continuation must
        # not swallow the next frame.
        conn = make_connection(PUBSUB_PUSH + bulk(b"next-command"))

        assert conn.read_response(push_request=True) == [
            b"message",
            b"ch",
            b"hello",
        ]
        assert conn.read_response() == b"next-command"


@pytest.mark.fixed_client
class TestDecodingFlagsSurvivePushInterleaving:
    def test_decode_responses_applies_after_push(self):
        # decode_responses=True must still decode the replies that bracket a
        # push; the push frame's bulk strings are decoded the same way.
        received = []
        conn = make_connection(
            bulk(SNOWMAN) + PUBSUB_PUSH + bulk(SNOWMAN),
            decode_responses=True,
        )
        conn._parser.set_pubsub_push_handler(
            lambda response: (received.append(response), response)[1]
        )

        assert conn.read_response() == "☃"
        assert conn.read_response() == "☃"
        assert received == [["message", "ch", "hello"]]

    def test_single_disable_decoding_survives_interleaved_push(self):
        # The per-call disable_decoding flag is forwarded through the push
        # continuation: both command reads here ask for raw bytes even though
        # the connection was created with decode_responses=True.
        conn = make_connection(
            bulk(SNOWMAN) + PUBSUB_PUSH + bulk(SNOWMAN),
            decode_responses=True,
        )

        assert conn.read_response(disable_decoding=True) == SNOWMAN
        assert conn.read_response(disable_decoding=True) == SNOWMAN

    def test_decoding_flag_can_change_per_call_across_push(self):
        conn = make_connection(
            bulk(SNOWMAN) + PUBSUB_PUSH + bulk(SNOWMAN),
            decode_responses=True,
        )

        assert conn.read_response() == "☃"
        assert conn.read_response(disable_decoding=True) == SNOWMAN


@pytest.mark.fixed_client
class TestPushHandlerFailure:
    def test_pubsub_handler_error_without_disconnect_is_rewound(self):
        # A failure in the pub/sub handler happens after the whole push frame
        # has been consumed, but the parser rewinds to the start of the read.
        # With disconnect_on_error=False the socket stays up; the same push is
        # therefore still queued and the next read fails identically instead
        # of handing a push to a command caller.
        conn = make_connection(PUBSUB_PUSH + bulk(b"second"))
        conn._parser.pubsub_push_handler_func = failing_handler

        with pytest.raises(PushHandlerError):
            conn.read_response(disconnect_on_error=False)
        assert conn.is_connected is True
        # The parser rewinds to the start of the failed read, so the push is
        # still the next frame regardless of how many socket reads fed it.
        assert conn._parser._buffer.get_pos() == 0
        assert conn._parser._buffer.unread_bytes() >= len(PUBSUB_PUSH)

        with pytest.raises(PushHandlerError):
            conn.read_response(disconnect_on_error=False)

    def test_pubsub_handler_error_disconnects_by_default(self):
        # The normal command path uses disconnect_on_error=True: a handler
        # failure tears the connection down so a poisoned stream never re-enters
        # the pool looking healthy.
        conn = make_connection(PUBSUB_PUSH + bulk(b"second"))
        conn._parser.pubsub_push_handler_func = failing_handler

        with pytest.raises(PushHandlerError):
            conn.read_response()
        assert conn.is_connected is False
        assert conn._parser._buffer is None

    def test_invalidation_handler_error_is_swallowed_and_connection_reused(self):
        # Invalidation/maintenance handlers are invoked inside
        # handle_push_response's own try/except: a failure is logged, the push
        # is considered handled and the following command reply is returned on
        # a still-healthy connection.
        conn = make_connection(INVALIDATION_PUSH + bulk(b"second"))
        conn._parser.set_invalidation_push_handler(failing_handler)

        assert conn.read_response(disconnect_on_error=False) == b"second"
        assert conn.is_connected is True
        assert conn._parser._buffer.unread_bytes() == 0


@pytest.mark.fixed_client
class TestPushInterleavingThroughConnectionAndPipeline:
    """End-to-end reads through Connection/pipeline, not just the parser.

    Every queued command gets a distinct sentinel reply so a push cannot be
    mistaken for one of them without the assertion catching it.
    """

    def _pipeline_connection(self, stream):
        conn = make_connection(stream, decode_responses=True)
        # The packed commands never leave the process; the canned stream
        # supplies every reply.
        conn.send_packed_command = mock.Mock()

        pool = mock.Mock(spec=ConnectionPool)
        pool.get_connection.return_value = conn
        pool.release = mock.Mock()

        client = Redis(protocol=3)
        client.connection_pool = pool
        return conn, client

    def test_pipeline_keeps_push_and_command_results_apart(self):
        invalidations = []
        pubsub_pushes = []
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
        conn._parser.set_pubsub_push_handler(
            lambda response: (pubsub_pushes.append(response), response)[1]
        )

        pipe = client.pipeline(transaction=False)
        pipe.get("a").get("b").get("c")
        assert pipe.execute() == ["sentinel-1", "sentinel-2", "sentinel-3"]

        assert invalidations == [["invalidate", ["foo"]]]
        assert pubsub_pushes == [["message", "ch", "hello"]]
        assert conn._parser._buffer.unread_bytes() == 0
        assert conn.is_connected is True

    def test_consecutive_pushes_inside_pipeline_dont_shift_results(self):
        pubsub_pushes = []
        stream = bulk(b"sentinel-1") + PUBSUB_PUSH + UNKNOWN_PUSH + bulk(b"sentinel-2")
        conn, client = self._pipeline_connection(stream)
        conn._parser.set_pubsub_push_handler(
            lambda response: (pubsub_pushes.append(response), response)[1]
        )

        pipe = client.pipeline(transaction=False)
        pipe.get("a").get("b")
        assert pipe.execute() == ["sentinel-1", "sentinel-2"]
        assert pubsub_pushes == [
            ["message", "ch", "hello"],
            ["custom-type", "payload"],
        ]
        assert conn._parser._buffer.unread_bytes() == 0
        assert conn.is_connected is True

    def test_connection_pair_of_commands_with_push_between(self):
        # The plain Connection command/response path (no pipeline object).
        received = []
        conn = make_connection(
            bulk(b"sentinel-1") + PUBSUB_PUSH + bulk(b"sentinel-2"),
            decode_responses=True,
        )
        conn._parser.set_pubsub_push_handler(
            lambda response: (received.append(response), response)[1]
        )
        conn.send_command = mock.Mock()

        first = conn.read_response()
        second = conn.read_response()
        assert (first, second) == ("sentinel-1", "sentinel-2")
        assert received == [["message", "ch", "hello"]]
        assert conn._parser._buffer.unread_bytes() == 0
