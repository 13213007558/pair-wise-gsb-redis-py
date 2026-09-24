import asyncio
import types

import pytest

from redis._parsers import (
    Encoder,
    _AsyncHiredisParser,
    _AsyncRESP2Parser,
    _AsyncRESP3Parser,
)
from redis._parsers.base import AttributedResponse
from redis.exceptions import RedisError, ResponseError
from redis.utils import HIREDIS_AVAILABLE


def make_parser(data: bytes, chunk_size: int = 65536) -> _AsyncRESP3Parser:
    """Feed a scripted byte stream to the parser through a real StreamReader,

    split into ``chunk_size`` pieces so frames are broken at arbitrary
    positions across stream reads.
    """
    stream = asyncio.StreamReader()
    for i in range(0, len(data), chunk_size):
        stream.feed_data(data[i : i + chunk_size])
    stream.feed_eof()
    parser = _AsyncRESP3Parser(socket_read_size=65536)
    connection = types.SimpleNamespace(
        _reader=stream, encoder=Encoder("utf-8", "strict", False)
    )
    parser.on_connect(connection)
    return parser


class PushRecorder:
    def __init__(self):
        self.messages = []

    async def __call__(self, response):
        self.messages.append(response)
        return response


ATTR_TTL = b"|1\r\n$3\r\nttl\r\n:100\r\n"
ATTR_FLAG = b"|1\r\n$4\r\nflag\r\n#t\r\n"
PUSH_MESSAGE = b">3\r\n$7\r\nmessage\r\n$7\r\nchannel\r\n$2\r\nhi\r\n"

# split the byte stream everywhere from byte-at-a-time to all-at-once
CHUNK_SIZES = [1, 2, 3, 5, 7, 64, 65536]


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_default_reply_consumes_attributes(chunk_size):
    parser = make_parser(ATTR_TTL + b"+OK\r\n", chunk_size)
    assert await parser.read_response() == b"OK"


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_opt_in_returns_attributes(chunk_size):
    parser = make_parser(ATTR_TTL + b"+OK\r\n", chunk_size)
    response = await parser.read_response(return_attributes=True)
    assert response == AttributedResponse(b"OK", {b"ttl": 100})


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_consecutive_attribute_frames_merge(chunk_size):
    parser = make_parser(ATTR_TTL + ATTR_FLAG + b"+OK\r\n", chunk_size)
    response = await parser.read_response(return_attributes=True)
    assert response == AttributedResponse(b"OK", {b"ttl": 100, b"flag": True})


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_nested_attributes_inside_aggregate(chunk_size):
    stream = b"*3\r\n:1\r\n" + ATTR_TTL + b":2\r\n:3\r\n"

    parser = make_parser(stream, chunk_size)
    assert await parser.read_response() == [1, 2, 3]

    parser = make_parser(stream, chunk_size)
    response = await parser.read_response(return_attributes=True)
    assert response.attributes is None
    assert response.value == [1, AttributedResponse(2, {b"ttl": 100}), 3]


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_attributes_are_not_mixed_into_reply_dict(chunk_size):
    stream = ATTR_TTL + b"%1\r\n$1\r\na\r\n:1\r\n"

    parser = make_parser(stream, chunk_size)
    # the metadata must not leak into the dict returned to the user
    assert await parser.read_response() == {b"a": 1}

    parser = make_parser(stream, chunk_size)
    response = await parser.read_response(return_attributes=True)
    assert response.value == {b"a": 1}
    assert response.attributes == {b"ttl": 100}


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_error_reply_with_attributes(chunk_size):
    stream = ATTR_TTL + b"-ERR unknown command\r\n"

    parser = make_parser(stream, chunk_size)
    response = await parser.read_response()
    assert isinstance(response, ResponseError)

    parser = make_parser(stream, chunk_size)
    response = await parser.read_response(return_attributes=True)
    assert isinstance(response.value, ResponseError)
    assert response.attributes == {b"ttl": 100}


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_push_between_attributes_and_value(chunk_size):
    recorder = PushRecorder()
    parser = make_parser(ATTR_TTL + PUSH_MESSAGE + b"+OK\r\n", chunk_size)
    parser.set_pubsub_push_handler(recorder)

    response = await parser.read_response(return_attributes=True)
    assert response == AttributedResponse(b"OK", {b"ttl": 100})
    assert recorder.messages == [[b"message", b"channel", b"hi"]]


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_push_does_not_consume_a_reply_slot(chunk_size):
    # a pipeline-style session: two commands, pushes interleaved; each
    # read_response must return the next command reply, never a push
    recorder = PushRecorder()
    stream = PUSH_MESSAGE + b":1\r\n" + PUSH_MESSAGE + b":2\r\n"
    parser = make_parser(stream, chunk_size)
    parser.set_pubsub_push_handler(recorder)

    assert await parser.read_response() == 1
    assert await parser.read_response() == 2
    assert recorder.messages == [[b"message", b"channel", b"hi"]] * 2


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_no_residual_attributes_on_next_command(chunk_size):
    parser = make_parser(ATTR_TTL + b"+OK\r\n+PONG\r\n", chunk_size)

    response = await parser.read_response(return_attributes=True)
    assert response.attributes == {b"ttl": 100}
    # the next command reads clean: no leftover attribute state
    response = await parser.read_response(return_attributes=True)
    assert response == AttributedResponse(b"PONG", None)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_interleaved_session(chunk_size):
    # normal results, attributes, pushes and errors interleaved on one
    # connection; every reply and its metadata must stay correctly attributed
    recorder = PushRecorder()
    stream = (
        ATTR_TTL
        + b"+OK\r\n"  # command 1: value with attributes
        + PUSH_MESSAGE  # out-of-band push
        + ATTR_FLAG
        + b"-ERR broken\r\n"  # command 2: error with attributes
        + b"*2\r\n"
        + ATTR_TTL
        + b"$3\r\nfoo\r\n:7\r\n"  # command 3: nested attribute
        + b"_\r\n"  # command 4: plain null, no attributes
    )
    parser = make_parser(stream, chunk_size)
    parser.set_pubsub_push_handler(recorder)

    response = await parser.read_response(return_attributes=True)
    assert response == AttributedResponse(b"OK", {b"ttl": 100})

    response = await parser.read_response(return_attributes=True)
    assert isinstance(response.value, ResponseError)
    assert response.attributes == {b"flag": True}

    response = await parser.read_response(return_attributes=True)
    assert response.attributes is None
    assert response.value == [AttributedResponse(b"foo", {b"ttl": 100}), 7]

    response = await parser.read_response(return_attributes=True)
    assert response == AttributedResponse(None, None)

    assert recorder.messages == [[b"message", b"channel", b"hi"]]


async def test_connection_read_response_from_parser_return_attributes():
    from redis.asyncio.connection import Connection

    conn = Connection(protocol=3)
    stream = asyncio.StreamReader()
    stream.feed_data(ATTR_TTL + b"+OK\r\n")
    stream.feed_eof()
    conn._parser.on_connect(
        types.SimpleNamespace(_reader=stream, encoder=Encoder("utf-8", "strict", False))
    )

    response = await conn._read_response_from_parser(return_attributes=True)
    assert response == AttributedResponse(b"OK", {b"ttl": 100})


async def test_resp2_parser_has_no_attributes_to_return():
    stream = asyncio.StreamReader()
    stream.feed_data(b"+OK\r\n")
    stream.feed_eof()
    parser = _AsyncRESP2Parser(socket_read_size=65536)
    parser.on_connect(
        types.SimpleNamespace(_reader=stream, encoder=Encoder("utf-8", "strict", False))
    )

    response = await parser.read_response(return_attributes=True)
    assert response == AttributedResponse(b"OK", None)


async def test_hiredis_parser_rejects_return_attributes():
    if not HIREDIS_AVAILABLE:
        pytest.skip("hiredis is not installed")
    parser = _AsyncHiredisParser(socket_read_size=65536)
    with pytest.raises(RedisError, match="attributes"):
        await parser.read_response(return_attributes=True)
