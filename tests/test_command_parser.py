from unittest.mock import Mock

import pytest
from redis._parsers import CommandsParser
from redis._parsers.commands import _extract_keys_from_specs
from redis.commands.metadata import RequestPolicy, ResponsePolicy
from redis.crc import key_slot
from redis.exceptions import ResponseError
from tests.helpers import get_expected_command_policies

from .conftest import (
    assert_resp_response,
    skip_if_redis_enterprise,
    skip_if_server_version_gte,
    skip_if_server_version_lt,
)


class TestCommandsParser:
    def test_init_commands(self, r):
        commands_parser = CommandsParser(r)
        assert commands_parser.commands is not None
        assert "get" in commands_parser.commands

    def test_get_keys_predetermined_key_location(self, r):
        commands_parser = CommandsParser(r)
        args1 = ["GET", "foo"]
        args2 = ["OBJECT", "encoding", "foo"]
        args3 = ["MGET", "foo", "bar", "foobar"]
        assert commands_parser.get_keys(r, *args1) == ["foo"]
        assert commands_parser.get_keys(r, *args2) == ["foo"]
        assert commands_parser.get_keys(r, *args3) == ["foo", "bar", "foobar"]

    @pytest.mark.filterwarnings("ignore:ResponseError")
    @skip_if_redis_enterprise()
    def test_get_moveable_keys(self, r):
        commands_parser = CommandsParser(r)
        args1 = [
            "EVAL",
            "return {KEYS[1],KEYS[2],ARGV[1],ARGV[2]}",
            2,
            "key1",
            "key2",
            "first",
            "second",
        ]
        args2 = ["XREAD", "COUNT", 2, b"STREAMS", "mystream", "writers", 0, 0]
        args3 = ["ZUNIONSTORE", "out", 2, "zset1", "zset2", "WEIGHTS", 2, 3]
        args4 = ["GEORADIUS", "Sicily", 15, 37, 200, "km", "WITHCOORD", b"STORE", "out"]
        args5 = ["MEMORY USAGE", "foo"]
        args6 = [
            "MIGRATE",
            "192.168.1.34",
            6379,
            "",
            0,
            5000,
            b"KEYS",
            "key1",
            "key2",
            "key3",
        ]
        args7 = ["MIGRATE", "192.168.1.34", 6379, "key1", 0, 5000]

        assert_resp_response(
            r,
            sorted(commands_parser.get_keys(r, *args1)),
            ["key1", "key2"],
            [b"key1", b"key2"],
        )
        assert_resp_response(
            r,
            sorted(commands_parser.get_keys(r, *args2)),
            ["mystream", "writers"],
            [b"mystream", b"writers"],
        )
        assert_resp_response(
            r,
            sorted(commands_parser.get_keys(r, *args3)),
            ["out", "zset1", "zset2"],
            [b"out", b"zset1", b"zset2"],
        )
        assert_resp_response(
            r,
            sorted(commands_parser.get_keys(r, *args4)),
            ["Sicily", "out"],
            [b"Sicily", b"out"],
        )
        assert sorted(commands_parser.get_keys(r, *args5)) in [["foo"], [b"foo"]]
        assert_resp_response(
            r,
            sorted(commands_parser.get_keys(r, *args6)),
            ["key1", "key2", "key3"],
            [b"key1", b"key2", b"key3"],
        )
        assert_resp_response(
            r, sorted(commands_parser.get_keys(r, *args7)), ["key1"], [b"key1"]
        )

    # A bug in redis<7.0 causes this to fail: https://github.com/redis/redis/issues/9493
    @skip_if_server_version_lt("7.0.0")
    def test_get_eval_keys_with_0_keys(self, r):
        commands_parser = CommandsParser(r)
        args = ["EVAL", "return {ARGV[1],ARGV[2]}", 0, "key1", "key2"]
        assert commands_parser.get_keys(r, *args) == []

    def test_get_pubsub_keys(self, r):
        commands_parser = CommandsParser(r)
        args1 = ["PUBLISH", "foo", "bar"]
        args2 = ["PUBSUB NUMSUB", "foo1", "foo2", "foo3"]
        args3 = ["PUBSUB channels", "*"]
        args4 = ["SUBSCRIBE", "foo1", "foo2", "foo3"]
        assert commands_parser.get_keys(r, *args1) == ["foo"]
        assert commands_parser.get_keys(r, *args2) == ["foo1", "foo2", "foo3"]
        assert commands_parser.get_keys(r, *args3) == ["*"]
        assert commands_parser.get_keys(r, *args4) == ["foo1", "foo2", "foo3"]

    @skip_if_server_version_lt("8.0.0")
    @skip_if_server_version_gte("8.5.240")
    @pytest.mark.onlycluster
    def test_get_command_policies(self, r):
        commands_parser = CommandsParser(r)

        expected_command_policies = get_expected_command_policies()

        actual_policies = commands_parser.get_command_policies()
        assert len(actual_policies) > 0

        for module_name, commands in expected_command_policies.items():
            for command, command_policies in commands.items():
                assert command in actual_policies[module_name]
                assert command_policies == [
                    command,
                    actual_policies[module_name][command].request_policy,
                    actual_policies[module_name][command].response_policy,
                ]

    @skip_if_server_version_lt("8.5.240")
    @pytest.mark.onlycluster
    def test_get_command_policies_json_debug_updated(self, r):
        commands_parser = CommandsParser(r)

        changes_in_defaults = {
            "json": {
                "debug": [
                    "debug",
                    RequestPolicy.DEFAULT_KEYLESS,
                    ResponsePolicy.DEFAULT_KEYLESS,
                ],
            },
        }

        expected_command_policies = get_expected_command_policies(changes_in_defaults)

        actual_policies = commands_parser.get_command_policies()
        assert len(actual_policies) > 0

        for module_name, commands in expected_command_policies.items():
            for command, command_policies in commands.items():
                assert command in actual_policies[module_name]
                assert command_policies == [
                    command,
                    actual_policies[module_name][command].request_policy,
                    actual_policies[module_name][command].response_policy,
                ]


def _to_resp2(value):
    """Convert a RESP3-shaped (map) key specification to the flat RESP2 shape."""
    if isinstance(value, dict):
        flattened = []
        for key, item in value.items():
            flattened.extend([key, _to_resp2(item)])
        return flattened
    if isinstance(value, list):
        return [_to_resp2(item) for item in value]
    return value


# Parametrizes a key-specification test over the two reply shapes a server can
# report the specs in: RESP3 maps and RESP2 flat name/value sequences.
RESP_SHAPES = pytest.mark.parametrize(
    "shape",
    [
        pytest.param(lambda value: value, id="resp3"),
        pytest.param(_to_resp2, id="resp2"),
    ],
)


def _index_spec(flags, index, lastkey=0, keystep=1, limit=0):
    return {
        "flags": flags,
        "begin_search": {"index": {"index": index}},
        "find_keys": {
            "range": {"lastkey": lastkey, "keystep": keystep, "limit": limit}
        },
    }


def _keyword_spec(flags, keyword, startfrom, lastkey=0, keystep=1, limit=0):
    return {
        "flags": flags,
        "begin_search": {"keyword": {"keyword": keyword, "startfrom": startfrom}},
        "find_keys": {
            "range": {"lastkey": lastkey, "keystep": keystep, "limit": limit}
        },
    }


def _keynum_spec(flags, index, keynum=0, firstkey=1, keystep=1):
    return {
        "flags": flags,
        "begin_search": {"index": {"index": index}},
        "find_keys": {
            "keynum": {"keynum": keynum, "firstkey": firstkey, "keystep": keystep}
        },
    }


def _command_entry(name, flags, key_specs=(), subcommands=()):
    """A parsed COMMAND reply entry, as CommandsParser.commands holds it."""
    return {
        "name": name,
        "arity": -1,
        "flags": flags,
        "first_key_pos": 0,
        "last_key_pos": 0,
        "step_count": 0,
        "key_specifications": list(key_specs),
        "subcommands": list(subcommands),
    }


def _subcommand_entry(name, flags, key_specs):
    """A raw subcommand entry of a COMMAND reply, as _parse_subcommand reads it."""
    return (name, -2, flags, 0, 0, 0, [], [], list(key_specs), [])


def _parser_for(commands):
    connection = Mock()
    connection.command.return_value = commands
    return CommandsParser(connection)


def _no_network_connection():
    """A connection that fails the test if any command is sent over it."""
    connection = Mock()
    connection.execute_command.side_effect = AssertionError(
        "key extraction issued a server round trip"
    )
    return connection


# key_slot(b"{k}1") - every key tagged {k} hashes to this slot.
TAGGED_SLOT = 7629


def _slot(key):
    return key_slot(key.encode() if isinstance(key, str) else key)


class TestLocalKeyExtraction:
    """
    Key extraction from Redis 7 key specifications, built from scripted COMMAND
    metadata - no server is involved, and the no-network connection proves that
    supported specifications are resolved without a COMMAND GETKEYS round trip.
    """

    @RESP_SHAPES
    def test_index_and_range_specs(self, shape):
        commands = {
            "bitop": _command_entry(
                "bitop",
                ["write", "movablekeys"],
                [
                    shape(_index_spec(["RW", "update"], 2)),
                    shape(_index_spec(["RO", "access"], 3, lastkey=-1)),
                ],
            )
        }
        parser = _parser_for(commands)
        args = ["BITOP", "AND", "{k}dest", "{k}1", "{k}2", "{k}3"]

        keys = parser.get_keys(_no_network_connection(), *args)

        assert keys == ["{k}dest", "{k}1", "{k}2", "{k}3"]
        assert {_slot(key) for key in keys} == {TAGGED_SLOT}

    @RESP_SHAPES
    def test_access_flags_are_preserved(self, shape):
        commands = {
            "bitop": _command_entry(
                "bitop",
                ["write", "movablekeys"],
                [
                    shape(_index_spec(["RW", "update"], 2)),
                    shape(_index_spec(["RO", "access"], 3, lastkey=-1)),
                ],
            )
        }
        args = ["BITOP", "AND", "{k}dest", "{k}1", "{k}2"]

        extracted = _extract_keys_from_specs(commands["bitop"], args)

        assert extracted == [
            ("{k}dest", {"RW", "update"}),
            ("{k}1", {"RO", "access"}),
            ("{k}2", {"RO", "access"}),
        ]

    @RESP_SHAPES
    def test_keyword_spec_and_optional_argument(self, shape):
        commands = {
            "georadius": _command_entry(
                "georadius",
                ["write", "movablekeys"],
                [
                    shape(_index_spec(["RO", "access"], 1)),
                    shape(_keyword_spec(["RW", "update"], "STORE", -1)),
                    shape(_keyword_spec(["RW", "update"], "STOREDIST", -1)),
                ],
            )
        }
        parser = _parser_for(commands)

        # The optional STORE argument group is absent: only the leading key.
        args = ["GEORADIUS", "Sicily", 15, 37, 200, "km", "WITHCOORD"]
        assert parser.get_keys(_no_network_connection(), *args) == ["Sicily"]

        # The keyword is matched case-insensitively, in any argument spelling.
        for store in ["STORE", "store", b"STORE", b"Store"]:
            args = ["GEORADIUS", "Sicily", 15, 37, 200, "km", store, "out"]
            keys = parser.get_keys(_no_network_connection(), *args)
            assert keys == ["Sicily", "out"]
            assert [_slot(key) for key in keys] == [10713, 16101]

    @RESP_SHAPES
    def test_keyword_search_from_the_end_with_limit(self, shape):
        commands = {
            "xread": _command_entry(
                "xread",
                ["readonly", "movablekeys"],
                [
                    shape(
                        _keyword_spec(
                            ["RO", "access"], "STREAMS", -1, lastkey=-1, limit=2
                        )
                    )
                ],
            )
        }
        parser = _parser_for(commands)

        args = ["XREAD", "COUNT", 2, "STREAMS", "mystream", "writers", 0, 0]
        keys = parser.get_keys(_no_network_connection(), *args)
        assert keys == ["mystream", "writers"]
        assert [_slot(key) for key in keys] == [12571, 14985]

        # Only the first half of the trailing arguments are streams (limit=2).
        args = ["XREAD", "STREAMS", "mystream", 0]
        assert parser.get_keys(_no_network_connection(), *args) == ["mystream"]

        # The keyword is searched from the end, exactly as the server does, so a
        # stream named like the keyword is taken for the keyword itself.
        args = ["XREAD", "STREAMS", "STREAMS", 0]
        assert parser.get_keys(_no_network_connection(), *args) == []

    @RESP_SHAPES
    def test_keynum_spec(self, shape):
        commands = {
            "eval": _command_entry(
                "eval",
                ["movablekeys"],
                [shape(_keynum_spec(["RW", "access", "update"], 2))],
            )
        }
        parser = _parser_for(commands)

        # The key counter is accepted as an int, a string or bytes.
        for numkeys in [2, "2", b"2"]:
            args = ["EVAL", "return 1", numkeys, "{k}1", "{k}2", "arg"]
            keys = parser.get_keys(_no_network_connection(), *args)
            assert keys == ["{k}1", "{k}2"]
            assert {_slot(key) for key in keys} == {TAGGED_SLOT}

        # Zero keys is a valid count.
        assert parser.get_keys(_no_network_connection(), *["EVAL", "return 1", 0]) == []

    @RESP_SHAPES
    def test_keynum_spec_with_keystep(self, shape):
        commands = {
            "cmd": _command_entry(
                "cmd",
                ["movablekeys"],
                [shape(_keynum_spec(["RO", "access"], 1, keystep=2))],
            )
        }
        parser = _parser_for(commands)

        args = ["CMD", 2, "{k}1", "skip", "{k}2", "skip"]
        keys = parser.get_keys(_no_network_connection(), *args)
        assert keys == ["{k}1", "{k}2"]
        assert {_slot(key) for key in keys} == {TAGGED_SLOT}

    @RESP_SHAPES
    def test_negative_begin_index(self, shape):
        commands = {
            "cmd": _command_entry(
                "cmd",
                ["movablekeys"],
                [shape(_index_spec(["RO", "access"], -1))],
            )
        }
        parser = _parser_for(commands)

        args = ["CMD", "not-a-key", "{k}1"]
        keys = parser.get_keys(_no_network_connection(), *args)
        assert keys == ["{k}1"]
        assert _slot(keys[0]) == TAGGED_SLOT

    @RESP_SHAPES
    @pytest.mark.parametrize(
        "args",
        [
            ["EVAL", "return 1", "not-a-number", "{k}1"],
            ["EVAL", "return 1", "2keys", "{k}1", "{k}2"],
            ["EVAL", "return 1", -1],
            ["EVAL", "return 1", 1.5, "{k}1"],
            ["EVAL", "return 1"],
            ["EVAL", "return 1", 3, "{k}1"],
        ],
        ids=[
            "non-integer",
            "trailing-junk",
            "negative",
            "float",
            "missing-counter",
            "more-keys-than-args",
        ],
    )
    def test_invalid_keynum_raises_before_sending(self, shape, args):
        commands = {
            "eval": _command_entry(
                "eval",
                ["movablekeys"],
                [shape(_keynum_spec(["RW", "access", "update"], 2))],
            )
        }
        parser = _parser_for(commands)

        # The malformed invocation is reported locally: the no-network connection
        # raises AssertionError if a command is sent over it.
        with pytest.raises(ResponseError, match="Invalid arguments"):
            parser.get_keys(_no_network_connection(), *args)

    @RESP_SHAPES
    def test_not_key_arguments_do_not_decide_the_slot(self, shape):
        commands = {
            "cmd": _command_entry(
                "cmd",
                ["movablekeys"],
                [
                    shape(_index_spec(["RO", "access"], 1)),
                    shape(_index_spec(["RO", "access", "not_key"], 2)),
                ],
            )
        }
        parser = _parser_for(commands)
        args = ["CMD", "{k}1", "not-a-key"]

        # The extraction preserves every argument with its flags...
        assert _extract_keys_from_specs(commands["cmd"], args) == [
            ("{k}1", {"RO", "access"}),
            ("not-a-key", {"RO", "access", "not_key"}),
        ]
        # ...but a not_key argument is not a key name and must not decide the slot.
        keys = parser.get_keys(_no_network_connection(), *args)
        assert keys == ["{k}1"]
        assert _slot(keys[0]) == TAGGED_SLOT

    @RESP_SHAPES
    def test_incomplete_spec_falls_back_to_server(self, shape):
        commands = {
            "migrate": _command_entry(
                "migrate",
                ["write", "movablekeys"],
                [
                    shape(_index_spec(["RW", "access", "incomplete"], 3)),
                    shape(_keyword_spec(["RW", "access"], "KEYS", -1, lastkey=-1)),
                ],
            )
        }
        parser = _parser_for(commands)
        connection = Mock()
        connection.execute_command.return_value = [b"key1", b"key2"]

        args = ["MIGRATE", "host", 6379, "", 0, 5000, "KEYS", "key1", "key2"]
        keys = parser.get_keys(connection, *args)

        assert keys == [b"key1", b"key2"]
        connection.execute_command.assert_called_once_with("COMMAND GETKEYS", *args)

    @pytest.mark.parametrize(
        "key_specs",
        [
            # An unknown begin_search mechanism.
            [
                {
                    "flags": ["RO"],
                    "begin_search": "unknown",
                    "find_keys": {"range": {"lastkey": 0, "keystep": 1, "limit": 0}},
                }
            ],
            # An unknown find_keys mechanism.
            [
                {
                    "flags": ["RO"],
                    "begin_search": {"index": {"index": 1}},
                    "find_keys": "unknown",
                }
            ],
            # A begin_search/find_keys shape this client does not implement.
            [
                {
                    "flags": ["RO"],
                    "begin_search": {"prefix": {}},
                    "find_keys": {"range": {"lastkey": 0, "keystep": 1, "limit": 0}},
                }
            ],
        ],
        ids=["unknown-begin-search", "unknown-find-keys", "unimplemented-shape"],
    )
    def test_unknown_spec_falls_back_to_server(self, key_specs):
        commands = {"cmd": _command_entry("cmd", ["movablekeys"], key_specs)}
        parser = _parser_for(commands)
        connection = Mock()
        connection.execute_command.return_value = [b"key1"]

        keys = parser.get_keys(connection, *["CMD", "key1"])

        assert keys == [b"key1"]
        connection.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "CMD", "key1"
        )

    def test_missing_key_specifications_fall_back_to_server(self):
        # A pre-7.0 server reports no key_specifications at all.
        entry = _command_entry("cmd", ["movablekeys"])
        del entry["key_specifications"]
        parser = _parser_for({"cmd": entry})
        connection = Mock()
        connection.execute_command.return_value = ["key1"]

        keys = parser.get_keys(connection, *["CMD", "key1"])

        assert keys == ["key1"]
        connection.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "CMD", "key1"
        )

    @RESP_SHAPES
    def test_subcommand_key_extraction(self, shape):
        commands = {
            "mycontainer": _command_entry(
                "mycontainer",
                ["container"],
                subcommands=[
                    _subcommand_entry(
                        "mycontainer|getall",
                        ["readonly", "movablekeys"],
                        [shape(_index_spec(["RO", "access"], 2))],
                    )
                ],
            )
        }
        parser = _parser_for(commands)

        # The subcommand is selected whether its name arrives as one argument
        # or as two.
        for args in [
            ["MYCONTAINER", "GETALL", "{k}1"],
            ["mycontainer getall", "{k}1"],
        ]:
            keys = parser.get_keys(_no_network_connection(), *args)
            assert keys == ["{k}1"]
            assert _slot(keys[0]) == TAGGED_SLOT
