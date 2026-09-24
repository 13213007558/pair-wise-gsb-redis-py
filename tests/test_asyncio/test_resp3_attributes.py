"""
Async tests for RESP3 attribute frame (|) support in _AsyncRESP3Parser.

A byte-level fake server (a real asyncio.StreamReader fed in chunks)
delivers a scripted reply stream that interleaves plain replies, attribute
frames, push messages and errors. Every reply is checked together with the
metadata attached to it, and the final plain reply proves no attribute
state leaks into the next command.
"""

import asyncio
import types

import pytest
from redis._parsers import Encoder, _AsyncRESP3Parser
from redis._parsers.base import AttributedResponse
from redis._parsers.hiredis import _AsyncHiredisParser
from redis.asyncio.connection import Connection
from redis.exceptions import RedisError, ResponseError

from tests.test_parsers.test_resp3_attributes import (
    CHUNK_SIZES,
    EXPECTED_PLAIN,
    SCRIPT,
    check_attributed,
    check_plain,
)


def make_parser() -> _AsyncRESP3Parser:
    stream = asyncio.StreamReader()
    parser = _AsyncRESP3Parser(socket_read_size=65536)
    connection = types.SimpleNamespace(
        _reader=stream,
        encoder=Encoder("utf-8", "strict", False),
    )
    parser.on_connect(connection)
    return parser, stream


def feed_script(stream, script, chunk_size):
    for pos in range(0, len(script), chunk_size):
        stream.feed_data(script[pos : pos + chunk_size])


async def read_script(parser, with_attributes):
    pushes = []

    async def record_push(response):
        pushes.append(response)

    parser.set_pubsub_push_handler(record_push)
    responses = [
        await parser.read_response(with_attributes=with_attributes)
        for _ in range(len(EXPECTED_PLAIN))
    ]
    return responses, pushes


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_resp3_attributes_dropped_by_default(chunk_size):
    parser, stream = make_parser()
    feed_script(stream, SCRIPT, chunk_size)
    responses, pushes = await read_script(parser, with_attributes=False)
    check_plain(responses, pushes)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
async def test_resp3_attributes_attached_when_opted_in(chunk_size):
    parser, stream = make_parser()
    feed_script(stream, SCRIPT, chunk_size)
    responses, pushes = await read_script(parser, with_attributes=True)
    check_attributed(responses, pushes)


async def test_resp3_attributes_do_not_leak_between_reads():
    # attribute frames attach only to the reply that immediately follows
    # them; the next command reads a clean response
    parser, stream = make_parser()
    feed_script(stream, SCRIPT, 4)
    for _ in range(len(EXPECTED_PLAIN) - 1):
        await parser.read_response()
    assert await parser.read_response() == b"PLAIN"

    parser, stream = make_parser()
    feed_script(stream, SCRIPT, 4)
    for _ in range(len(EXPECTED_PLAIN) - 1):
        await parser.read_response(with_attributes=True)
    last = await parser.read_response(with_attributes=True)
    assert last == AttributedResponse(b"PLAIN", {})


async def test_resp3_attributes_opt_in_wraps_error_without_raising():
    parser, stream = make_parser()
    feed_script(stream, b"|1\r\n$3\r\nerr\r\n:9\r\n-ERR boom\r\n", 3)
    response = await parser.read_response(with_attributes=True)
    assert isinstance(response, AttributedResponse)
    assert isinstance(response.value, ResponseError)
    assert str(response.value) == "boom"
    assert response.attributes == {b"err": 9}


async def test_async_hiredis_read_response_with_attributes_raises():
    parser = _AsyncHiredisParser.__new__(_AsyncHiredisParser)
    with pytest.raises(RedisError, match="not supported by the hiredis parser"):
        await parser.read_response(with_attributes=True)


async def test_connection_read_response_passes_with_attributes():
    conn = Connection(protocol=3)
    parser, stream = make_parser()
    feed_script(stream, b"|1\r\n$3\r\nfoo\r\n$3\r\nbar\r\n+OK\r\n", 2)
    conn._parser = parser

    response = await conn.read_response(with_attributes=True)
    assert response == AttributedResponse(b"OK", {b"foo": b"bar"})


async def test_connection_read_response_with_attributes_returns_error_in_wrapper():
    conn = Connection(protocol=3)
    parser, stream = make_parser()
    feed_script(stream, b"|1\r\n$3\r\nerr\r\n:9\r\n-ERR boom\r\n", 3)
    conn._parser = parser

    response = await conn.read_response(with_attributes=True)
    assert isinstance(response, AttributedResponse)
    assert isinstance(response.value, ResponseError)
    assert response.attributes == {b"err": 9}
