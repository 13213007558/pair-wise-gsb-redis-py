"""
Tests for RESP3 attribute frame (|) support in the pure-Python parsers.

A byte-level fake server feeds a scripted reply stream to the parser in
chunks, so frames are split at arbitrary positions. The script interleaves
plain replies, attribute frames, push messages and errors; every reply is
checked together with the metadata attached to it, and the final plain
reply proves no attribute state leaks into the next command.
"""

import types

import pytest
from redis._parsers import Encoder, _RESP3Parser
from redis._parsers.base import AttributedResponse
from redis._parsers.hiredis import _HiredisParser
from redis.connection import Connection
from redis.exceptions import RedisError, ResponseError


class ChunkedSocket:
    """A fake server socket that serves a scripted byte stream in chunks.

    Every ``recv`` returns at most ``chunk_size`` bytes, so the parser sees
    frames split at arbitrary positions.
    """

    def __init__(self, data: bytes, chunk_size: int):
        self._data = data
        self._chunk_size = chunk_size
        self._pos = 0

    def recv(self, bufsize):
        if self._pos >= len(self._data):
            return b""
        end = min(self._pos + min(bufsize, self._chunk_size), len(self._data))
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def settimeout(self, timeout):
        pass


# The scripted server replies, one per "command":
#   1. simple string preceded by an attribute frame
#   2. aggregate with attribute frames on nested elements
#   3. two consecutive attribute frames before one reply
#   4. a push message interleaved between the attribute frame and its reply
#   5. an error reply preceded by an attribute frame
#   6. a map with an attribute frame on one of its values
#   7. a plain reply: proves no attribute state leaks into the next command
SCRIPT = (
    b"|1\r\n$3\r\nfoo\r\n$3\r\nbar\r\n+OK\r\n"
    b"*3\r\n"
    b"|1\r\n$2\r\nk1\r\n:1\r\n+one\r\n"
    b"+two\r\n"
    b"|1\r\n$2\r\nk3\r\n:3\r\n*2\r\n+a\r\n+b\r\n"
    b"|1\r\n$3\r\nfoo\r\n:1\r\n|1\r\n$3\r\nbar\r\n:2\r\n+MERGED\r\n"
    b"|1\r\n$2\r\nmk\r\n$2\r\nmv\r\n"
    b">3\r\n$7\r\nmessage\r\n$2\r\nch\r\n$5\r\nhello\r\n"
    b"+AFTER_PUSH\r\n"
    b"|1\r\n$3\r\nerr\r\n:9\r\n-ERR boom\r\n"
    b"%2\r\n$1\r\na\r\n|1\r\n$2\r\nvk\r\n:5\r\n$1\r\nb\r\n$1\r\nc\r\n:7\r\n"
    b"+PLAIN\r\n"
)

EXPECTED_PLAIN = [
    b"OK",
    [b"one", b"two", [b"a", b"b"]],
    b"MERGED",
    b"AFTER_PUSH",
    "boom",  # placeholder for index 4: a ResponseError instance
    {b"a": b"b", b"c": 7},
    b"PLAIN",
]

EXPECTED_ATTRIBUTED = [
    AttributedResponse(b"OK", {b"foo": b"bar"}),
    AttributedResponse(
        [
            AttributedResponse(b"one", {b"k1": 1}),
            b"two",
            AttributedResponse([b"a", b"b"], {b"k3": 3}),
        ],
        {},
    ),
    AttributedResponse(b"MERGED", {b"foo": 1, b"bar": 2}),
    AttributedResponse(b"AFTER_PUSH", {b"mk": b"mv"}),
    AttributedResponse("boom", {b"err": 9}),  # placeholder: value is an error
    AttributedResponse({b"a": AttributedResponse(b"b", {b"vk": 5}), b"c": 7}, {}),
    AttributedResponse(b"PLAIN", {}),
]

CHUNK_SIZES = [1, 2, 3, 5, 7, 64, len(SCRIPT)]


def make_parser(data: bytes, chunk_size: int) -> _RESP3Parser:
    parser = _RESP3Parser(socket_read_size=65536)
    connection = types.SimpleNamespace(
        _sock=ChunkedSocket(data, chunk_size),
        socket_timeout=None,
        encoder=Encoder("utf-8", "strict", False),
    )
    parser.on_connect(connection)
    return parser


def read_script(parser, with_attributes):
    pushes = []
    parser.set_pubsub_push_handler(lambda response: pushes.append(response))
    responses = [
        parser.read_response(with_attributes=with_attributes)
        for _ in range(len(EXPECTED_PLAIN))
    ]
    return responses, pushes


def check_plain(responses, pushes):
    for i, (response, expected) in enumerate(zip(responses, EXPECTED_PLAIN)):
        if i == 4:
            assert isinstance(response, ResponseError)
            assert str(response) == "boom"
        else:
            assert response == expected
        # no attribute wrapper may leak into the default response shape
        if isinstance(response, list):
            assert not any(isinstance(item, AttributedResponse) for item in response)
    # the interleaved push went to the push handler and did not take up
    # a response slot
    assert pushes == [[b"message", b"ch", b"hello"]]


def check_attributed(responses, pushes):
    for i, (response, expected) in enumerate(zip(responses, EXPECTED_ATTRIBUTED)):
        assert isinstance(response, AttributedResponse)
        if i == 4:
            assert isinstance(response.value, ResponseError)
            assert str(response.value) == "boom"
            assert response.attributes == expected.attributes
        else:
            assert response == expected
    assert pushes == [[b"message", b"ch", b"hello"]]


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_resp3_attributes_dropped_by_default(chunk_size):
    parser = make_parser(SCRIPT, chunk_size)
    responses, pushes = read_script(parser, with_attributes=False)
    check_plain(responses, pushes)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_resp3_attributes_attached_when_opted_in(chunk_size):
    parser = make_parser(SCRIPT, chunk_size)
    responses, pushes = read_script(parser, with_attributes=True)
    check_attributed(responses, pushes)


def test_resp3_attributes_do_not_leak_between_reads():
    # attribute frames attach only to the reply that immediately follows
    # them; the next command reads a clean response
    parser = make_parser(SCRIPT, chunk_size=4)
    *_, last = (parser.read_response() for _ in range(len(EXPECTED_PLAIN)))
    assert last == b"PLAIN"

    parser = make_parser(SCRIPT, chunk_size=4)
    *_, last = (
        parser.read_response(with_attributes=True) for _ in range(len(EXPECTED_PLAIN))
    )
    assert last == AttributedResponse(b"PLAIN", {})


def test_resp3_attributes_opt_in_wraps_error_without_raising():
    parser = make_parser(b"|1\r\n$3\r\nerr\r\n:9\r\n-ERR boom\r\n", 3)
    response = parser.read_response(with_attributes=True)
    assert isinstance(response, AttributedResponse)
    assert isinstance(response.value, ResponseError)
    assert str(response.value) == "boom"
    assert response.attributes == {b"err": 9}


def test_hiredis_read_response_with_attributes_raises():
    parser = _HiredisParser.__new__(_HiredisParser)
    with pytest.raises(RedisError, match="not supported by the hiredis parser"):
        parser.read_response(with_attributes=True)


def test_connection_read_response_passes_with_attributes():
    conn = Connection(protocol=3)
    parser = make_parser(b"|1\r\n$3\r\nfoo\r\n$3\r\nbar\r\n+OK\r\n", chunk_size=2)
    conn._parser = parser

    response = conn.read_response(with_attributes=True)
    assert response == AttributedResponse(b"OK", {b"foo": b"bar"})


def test_connection_read_response_with_attributes_returns_error_in_wrapper():
    conn = Connection(protocol=3)
    parser = make_parser(b"|1\r\n$3\r\nerr\r\n:9\r\n-ERR boom\r\n", chunk_size=3)
    conn._parser = parser

    response = conn.read_response(with_attributes=True)
    assert isinstance(response, AttributedResponse)
    assert isinstance(response.value, ResponseError)
    assert response.attributes == {b"err": 9}
