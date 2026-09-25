from unittest import mock

import pytest
from redis._parsers import AttributedResponse, _HiredisParser, _RESP3Parser
from redis._parsers.encoders import Encoder
from redis.connection import Connection
from redis.exceptions import RedisError
from redis.utils import HIREDIS_AVAILABLE
from tests.mocks import MockSocket


def make_parser(data=b"", response_attributes=False):
    """Build a RESP3 parser reading from a fake socket.

    MockSocket truncates every read to 5 bytes, so replies are unpacked
    chunk by chunk rather than in a single read.
    """
    parser = _RESP3Parser(socket_read_size=65536)
    connection = mock.Mock()
    connection._sock = MockSocket(data)
    connection.socket_timeout = None
    connection.encoder = Encoder("utf-8", "strict", False)
    parser.on_connect(connection)
    parser.response_attributes_enabled = response_attributes
    return parser


def test_attributes_are_discarded_by_default():
    parser = make_parser(b"|1\r\n+ttl\r\n:3600\r\n+OK\r\n")

    assert parser.read_response() == b"OK"


def test_attributes_are_returned_when_enabled():
    parser = make_parser(b"|1\r\n+ttl\r\n:3600\r\n+OK\r\n", True)

    assert parser.read_response() == AttributedResponse(b"OK", {b"ttl": 3600})


def test_consecutive_attribute_sections_are_merged():
    # later sections override keys of earlier ones
    parser = make_parser(
        b"|1\r\n+a\r\n:1\r\n|2\r\n+b\r\n:2\r\n+a\r\n:9\r\n+OK\r\n", True
    )

    assert parser.read_response() == AttributedResponse(b"OK", {b"a": 9, b"b": 2})


def test_attributes_on_nested_aggregate_elements():
    data = (
        b"|1\r\n+ttl\r\n:100\r\n"
        b"%2\r\n"
        b"+k1\r\n"
        b"|1\r\n+m\r\n:1\r\n*2\r\n+a\r\n+b\r\n"
        b"+k2\r\n"
        b":5\r\n"
    )
    parser = make_parser(data, True)

    assert parser.read_response() == AttributedResponse(
        {
            b"k1": AttributedResponse([b"a", b"b"], {b"m": 1}),
            b"k2": 5,
        },
        {b"ttl": 100},
    )


def test_attributes_on_nested_aggregate_elements_disabled():
    data = (
        b"|1\r\n+ttl\r\n:100\r\n"
        b"%2\r\n"
        b"+k1\r\n"
        b"|1\r\n+m\r\n:1\r\n*2\r\n+a\r\n+b\r\n"
        b"+k2\r\n"
        b":5\r\n"
    )
    parser = make_parser(data)

    assert parser.read_response() == {b"k1": [b"a", b"b"], b"k2": 5}


def test_attributes_on_scalar_values():
    parser = make_parser(
        b"|1\r\n+a\r\n:1\r\n_\r\n|1\r\n+a\r\n:2\r\n#t\r\n|1\r\n+a\r\n:3\r\n,1.5\r\n",
        True,
    )

    assert parser.read_response() == AttributedResponse(None, {b"a": 1})
    assert parser.read_response() == AttributedResponse(True, {b"a": 2})
    assert parser.read_response() == AttributedResponse(1.5, {b"a": 3})


def test_attributes_before_error_reply_are_discarded():
    parser = make_parser(b"|1\r\n+a\r\n:1\r\n-ERR boom\r\n", True)

    response = parser.read_response()
    assert isinstance(response, Exception)
    assert str(response) == "boom"


def test_push_attributes_are_handled_by_the_push_handler():
    data = (
        b"|1\r\n+push-attr\r\n:7\r\n>3\r\n+message\r\n+ch\r\n+hello\r\n"
        b"|1\r\n+cmd-attr\r\n:2\r\n+OK\r\n"
    )
    parser = make_parser(data, True)
    pushes = []
    parser.set_pubsub_push_handler(pushes.append)

    # the push message keeps its own attributes and does not leak them
    # into the command reply that follows it
    assert parser.read_response() == AttributedResponse(b"OK", {b"cmd-attr": 2})
    assert pushes == [
        AttributedResponse([b"message", b"ch", b"hello"], {b"push-attr": 7})
    ]


def test_push_attributes_are_discarded_by_default():
    data = b"|1\r\n+push-attr\r\n:7\r\n>3\r\n+message\r\n+ch\r\n+hello\r\n+OK\r\n"
    parser = make_parser(data)
    pushes = []
    parser.set_pubsub_push_handler(pushes.append)

    assert parser.read_response() == b"OK"
    assert pushes == [[b"message", b"ch", b"hello"]]


def test_push_request_returns_push_with_its_attributes():
    parser = make_parser(
        b"|1\r\n+push-attr\r\n:7\r\n>3\r\n+message\r\n+ch\r\n+hello\r\n",
        True,
    )

    assert parser.read_response(push_request=True) == AttributedResponse(
        [b"message", b"ch", b"hello"], {b"push-attr": 7}
    )


def test_mixed_stream_settles_exactly_two_pipeline_replies():
    # a fake peer mixing attributes, a push message, a nested map and two
    # command replies: exactly two read_response() calls settle, and the
    # whole stream is consumed by them
    data = (
        b"|1\r\n+ttl\r\n:100\r\n"
        b"%2\r\n"
        b"+k1\r\n"
        b"|1\r\n+m\r\n:1\r\n*2\r\n+a\r\n+b\r\n"
        b"+k2\r\n"
        b":5\r\n"
        b"|1\r\n+push-attr\r\n:7\r\n>3\r\n+message\r\n+ch\r\n+hello\r\n"
        b"|1\r\n+ttl\r\n:200\r\n|1\r\n+x\r\n:1\r\n+OK\r\n"
    )
    parser = make_parser(data, True)
    pushes = []
    parser.set_pubsub_push_handler(pushes.append)

    first = parser.read_response()
    second = parser.read_response()

    assert first == AttributedResponse(
        {
            b"k1": AttributedResponse([b"a", b"b"], {b"m": 1}),
            b"k2": 5,
        },
        {b"ttl": 100},
    )
    assert second == AttributedResponse(b"OK", {b"ttl": 200, b"x": 1})
    assert pushes == [
        AttributedResponse([b"message", b"ch", b"hello"], {b"push-attr": 7})
    ]
    # both replies (and the interleaved push) consumed the whole stream:
    # the attribute sections were not settled as replies of their own
    assert parser._buffer._sock.pos == len(data)


def test_hiredis_parser_rejects_response_attributes():
    parser = _HiredisParser.__new__(_HiredisParser)

    assert parser.response_attributes_enabled is False
    with pytest.raises(RedisError, match="response attributes"):
        parser.response_attributes_enabled = True


def test_connection_opts_parser_into_response_attributes():
    conn = Connection(protocol=3, parser_class=_RESP3Parser, response_attributes=True)

    assert conn._parser.response_attributes_enabled is True


@pytest.mark.skipif(not HIREDIS_AVAILABLE, reason="hiredis is not installed")
def test_connection_response_attributes_requires_python_parser_with_hiredis():
    with pytest.raises(RedisError, match="response attributes"):
        Connection(protocol=3, response_attributes=True)
