import socket
from unittest import mock

import pytest

from redis._parsers import AttributedValue, _HiredisParser, _RESP3Parser
from redis._parsers.encoders import Encoder
from redis.connection import Connection
from redis.exceptions import RedisError, ResponseError
from redis.utils import HIREDIS_AVAILABLE


def frame(text: str) -> bytes:
    """Encode a readable RESP stream: each \n becomes a CRLF terminator."""
    return text.replace("\n", "\r\n").encode()


@pytest.fixture
def make_parser():
    """Build _RESP3Parsers wired to one end of a local fake RESP peer."""
    parsers_and_peers = []

    def factory(socket_read_size=65536, expose_attributes=False):
        client, server = socket.socketpair()
        parser = _RESP3Parser(socket_read_size=socket_read_size)
        parser.expose_attributes = expose_attributes
        connection = mock.Mock()
        connection._sock = client
        connection.socket_timeout = None
        connection.encoder = Encoder("utf-8", "strict", False)
        parser.on_connect(connection)
        parsers_and_peers.append((parser, server))
        return parser, server

    yield factory

    for parser, server in parsers_and_peers:
        parser.on_disconnect()
        server.close()


def test_attributes_are_discarded_by_default(make_parser):
    parser, server = make_parser()
    server.sendall(frame("|1\n$3\nfoo\n$3\nbar\n+OK\n"))

    assert parser.read_response() == b"OK"


def test_expose_attributes_wraps_reply(make_parser):
    parser, server = make_parser(expose_attributes=True)
    server.sendall(frame("|1\n$3\nfoo\n$3\nbar\n+OK\n"))

    response = parser.read_response()
    assert response == AttributedValue(b"OK", {b"foo": b"bar"})
    assert response.value == b"OK"
    assert response.attributes == {b"foo": b"bar"}


def test_consecutive_attribute_frames_merge_with_later_keys_winning(make_parser):
    parser, server = make_parser(expose_attributes=True)
    server.sendall(frame("|1\n$1\na\n:1\n|2\n$1\nb\n:2\n$1\na\n:3\n:42\n"))

    response = parser.read_response()
    assert response == AttributedValue(42, {b"a": 3, b"b": 2})


def test_attributes_on_nested_elements_keep_their_position(make_parser):
    parser, server = make_parser(expose_attributes=True)
    # the attribute frame modifies the array that is the map value of "key1"
    server.sendall(frame("%1\n$4\nkey1\n|1\n$3\nfoo\n$3\nbar\n*2\n:1\n:2\n"))

    response = parser.read_response()
    assert response == {b"key1": AttributedValue([1, 2], {b"foo": b"bar"})}


def test_attributes_preceding_an_error_do_not_change_its_shape(make_parser):
    parser, server = make_parser(expose_attributes=True)
    server.sendall(frame("|1\n$3\nfoo\n$3\nbar\n-ERR broken\n"))

    response = parser.read_response()
    assert isinstance(response, ResponseError)
    assert not isinstance(response, AttributedValue)


def test_push_attributes_go_to_the_push_handler(make_parser):
    parser, server = make_parser(expose_attributes=True)
    parser.handle_push_response = mock.Mock()
    server.sendall(
        frame(
            "|1\n$4\nmeta\n:9\n>2\n$10\ninvalidate\n*1\n$3\nkey\n"
            "|1\n$3\nbaz\n:4\n$2\nok\n"
        )
    )

    response = parser.read_response()

    # the push message got its own attributes...
    parser.handle_push_response.assert_called_once_with(
        [b"invalidate", [b"key"]], attributes={b"meta": 9}
    )
    # ...and did not leak them into the regular command response
    assert response == AttributedValue(b"ok", {b"baz": 4})


def test_pipeline_with_attributes_push_and_nested_maps_settles_two_replies(make_parser):
    # a fake peer mixes attributes, a push message and nested maps into the
    # replies of a two-command pipeline; read byte-by-byte (socket_read_size=1)
    # exactly two command results may settle.
    parser, server = make_parser(socket_read_size=1, expose_attributes=True)
    invalidation_handler = mock.Mock()
    parser.invalidation_push_handler_func = invalidation_handler
    server.sendall(
        frame(
            # reply 1: attributes + nested map
            "|1\n$3\nfoo\n$3\nbar\n%1\n$1\na\n*2\n:1\n:2\n"
            # an interleaved push message carrying its own attributes
            "|1\n$4\nmeta\n:9\n>2\n$10\ninvalidate\n*1\n$3\nkey\n"
            # reply 2: attributes + bulk string
            "|1\n$3\nbaz\n:4\n$2\nok\n"
        )
    )

    first = parser.read_response()
    second = parser.read_response()

    assert first == AttributedValue({b"a": [1, 2]}, {b"foo": b"bar"})
    assert second == AttributedValue(b"ok", {b"baz": 4})
    invalidation_handler.assert_called_once_with([b"invalidate", [b"key"]])
    # the stream is fully settled: nothing is left to read
    assert parser.can_read(timeout=0) is False


def test_expose_attributes_requires_resp3():
    with pytest.raises(RedisError, match="only supported with RESP version 3"):
        Connection(expose_attributes=True, protocol=2)


def test_expose_attributes_is_forwarded_to_the_resp3_parser():
    connection = Connection(expose_attributes=True, protocol=3)
    assert isinstance(connection._parser, _RESP3Parser)
    assert connection._parser.expose_attributes is True

    connection = Connection(protocol=3)
    assert connection._parser.expose_attributes is False


@pytest.mark.skipif(not HIREDIS_AVAILABLE, reason="hiredis is not installed")
def test_expose_attributes_is_rejected_with_the_hiredis_parser():
    # hiredis does not surface RESP3 attributes; opting in must report the
    # requirement instead of silently dropping the metadata.
    with pytest.raises(RedisError, match="pure-Python RESP3 parser"):
        Connection(expose_attributes=True, protocol=3, parser_class=_HiredisParser)
