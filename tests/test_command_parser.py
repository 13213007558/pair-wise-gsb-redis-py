from unittest.mock import Mock

import pytest
from redis._parsers import CommandsParser
from redis._parsers.commands import _get_keys_from_key_specs
from redis.commands.metadata import RequestPolicy, ResponsePolicy
from tests.helpers import get_expected_command_policies, movablekeys_commands_table

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
        """
        Commands whose keys cannot be interpreted locally are still resolved by
        `COMMAND GETKEYS` (MIGRATE's key specifications are incomplete) or carry
        predetermined key positions (MEMORY USAGE).
        """
        commands_parser = CommandsParser(r)
        args1 = ["MEMORY USAGE", "foo"]
        args2 = [
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
        args3 = ["MIGRATE", "192.168.1.34", 6379, "key1", 0, 5000]

        assert sorted(commands_parser.get_keys(r, *args1)) in [["foo"], [b"foo"]]
        assert_resp_response(
            r,
            sorted(commands_parser.get_keys(r, *args2)),
            ["key1", "key2", "key3"],
            [b"key1", b"key2", b"key3"],
        )
        assert_resp_response(
            r, sorted(commands_parser.get_keys(r, *args3)), ["key1"], [b"key1"]
        )

    @skip_if_server_version_lt("7.0.0")
    @skip_if_redis_enterprise()
    def test_get_movablekeys_from_key_specs(self, r):
        """
        movablekeys commands with complete key specifications are interpreted
        locally, so the keys are the invocation's own arguments whatever the
        protocol, and no COMMAND GETKEYS round trip is needed.
        """
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

        assert sorted(commands_parser.get_keys(r, *args1)) == ["key1", "key2"]
        assert sorted(commands_parser.get_keys(r, *args2)) == ["mystream", "writers"]
        assert sorted(commands_parser.get_keys(r, *args3)) == ["out", "zset1", "zset2"]
        assert sorted(commands_parser.get_keys(r, *args4)) == ["Sicily", "out"]

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


class TestKeySpecInterpretation:
    """
    Local interpretation of `movablekeys` commands' key specifications.

    The parser is built over a synthetic `COMMAND` reply, so no server is
    needed; a mock connection stands in for the `COMMAND GETKEYS` fallback and
    records whether it was consulted.
    """

    @staticmethod
    def parser(getkeys_return=None):
        connection = Mock()
        connection.command.return_value = {}
        connection.execute_command.return_value = getkeys_return or []
        parser = CommandsParser(connection)
        parser.commands = movablekeys_commands_table()
        return parser, connection

    def test_xread_keyword_begin_search_and_limit(self):
        # The keys after STREAMS are half of the remaining arguments; the IDs
        # and the COUNT option must not be mistaken for keys.
        parser, connection = self.parser()
        args = ["XREAD", "COUNT", 2, "STREAMS", "mystream", "writers", 0, 0]
        assert parser.get_keys(connection, *args) == ["mystream", "writers"]
        connection.execute_command.assert_not_called()

    def test_keyword_begin_search_matches_bytes_and_case_insensitively(self):
        parser, connection = self.parser()
        args = ["XREAD", b"COUNT", 2, b"streams", b"mystream", 0]
        assert parser.get_keys(connection, *args) == [b"mystream"]
        connection.execute_command.assert_not_called()

    def test_xread_without_keys(self):
        # STREAMS is the last argument, so the key range is empty.
        parser, connection = self.parser()
        assert parser.get_keys(connection, "XREAD", "STREAMS") == []

    def test_zunion_keynum(self):
        parser, connection = self.parser()
        args = ["ZUNION", 2, "zset1", "zset2", "WEIGHTS", 2, 3]
        assert parser.get_keys(connection, *args) == ["zset1", "zset2"]
        connection.execute_command.assert_not_called()

    def test_zunionstore_multiple_specs(self):
        # The destination comes from the first spec, the sources from the
        # keynum spec.
        parser, connection = self.parser()
        args = ["ZUNIONSTORE", "out", 2, "zset1", "zset2", "WEIGHTS", 2, 3]
        assert parser.get_keys(connection, *args) == ["out", "zset1", "zset2"]
        connection.execute_command.assert_not_called()

    def test_keynum_read_from_bytes_argument(self):
        parser, connection = self.parser()
        args = ["ZUNIONSTORE", "out", b"2", "zset1", "zset2"]
        assert parser.get_keys(connection, *args) == ["out", "zset1", "zset2"]

    def test_keynum_zero_keys(self):
        parser, connection = self.parser()
        args = ["EVAL", "return {ARGV[1]}", 0, "key1"]
        assert parser.get_keys(connection, *args) == []

    def test_keynum_beyond_the_arguments_falls_back_to_getkeys(self):
        # The declared key count exceeds the argument list, so the invocation
        # cannot be interpreted locally.
        parser, connection = self.parser(getkeys_return=["zset1"])
        args = ["ZUNIONSTORE", "out", 5, "zset1"]
        assert parser.get_keys(connection, *args) == ["zset1"]
        connection.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "ZUNIONSTORE", "out", 5, "zset1"
        )

    def test_georadius_optional_keyword_specs(self):
        parser, connection = self.parser()
        args = ["GEORADIUS", "Sicily", 15, 37, 200, "km", "WITHCOORD", "STORE", "out"]
        assert parser.get_keys(connection, *args) == ["Sicily", "out"]
        connection.execute_command.assert_not_called()

    def test_georadius_without_optional_keywords(self):
        # Neither STORE nor STOREDIST is present, so only the index spec matches.
        parser, connection = self.parser()
        args = ["GEORADIUS", "Sicily", 15, 37, 200, "km", "WITHCOORD"]
        assert parser.get_keys(connection, *args) == ["Sicily"]

    def test_range_with_keystep(self):
        parser, connection = self.parser()
        args = ["KEYSTEP.CMD", "key1", "value1", "key2", "value2"]
        assert parser.get_keys(connection, *args) == ["key1", "key2"]
        connection.execute_command.assert_not_called()

    def test_not_key_spec_contributes_no_keys(self):
        # The argument is a shard channel, not a key name: it must not be
        # routed by.
        parser, connection = self.parser()
        assert parser.get_keys(connection, "SHARD.PUB", "channel") == []
        connection.execute_command.assert_not_called()

    def test_incomplete_spec_falls_back_to_getkeys(self):
        parser, connection = self.parser(getkeys_return=["key1"])
        assert parser.get_keys(connection, "INCOMPLETE.CMD", "key1") == ["key1"]
        connection.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "INCOMPLETE.CMD", "key1"
        )

    def test_variable_flags_spec_falls_back_to_getkeys(self):
        parser, connection = self.parser(getkeys_return=["key1"])
        assert parser.get_keys(connection, "VARFLAGS.CMD", "key1") == ["key1"]
        connection.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "VARFLAGS.CMD", "key1"
        )

    def test_unknown_begin_search_falls_back_to_getkeys(self):
        parser, connection = self.parser(getkeys_return=["key1"])
        assert parser.get_keys(connection, "UNKNOWNBEGIN.CMD", "key1") == ["key1"]
        connection.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "UNKNOWNBEGIN.CMD", "key1"
        )

    def test_unknown_find_keys_falls_back_to_getkeys(self):
        parser, connection = self.parser(getkeys_return=["key1"])
        assert parser.get_keys(connection, "UNKNOWNFIND.CMD", "key1") == ["key1"]
        connection.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "UNKNOWNFIND.CMD", "key1"
        )

    def test_missing_key_specifications_fall_back_to_getkeys(self):
        # A server too old to report key specifications.
        parser, connection = self.parser(getkeys_return=["key1"])
        entry = movablekeys_commands_table()["xread"]
        del entry["key_specifications"]
        parser.commands["xread"] = entry
        args = ["XREAD", "STREAMS", "mystream", 0]
        assert parser.get_keys(connection, *args) == ["key1"]
        connection.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "XREAD", "STREAMS", "mystream", 0
        )

    def test_movablekeys_subcommand_of_module_command(self):
        parser, connection = self.parser()
        args = ["MODULE.CMD SUB", "key1", "key2"]
        assert parser.get_keys(connection, *args) == ["key1", "key2"]
        connection.execute_command.assert_not_called()

    def test_movablekeys_subcommand_falls_back_to_getkeys(self):
        parser, connection = self.parser(getkeys_return=["key1"])
        args = ["MODULE.CMD PARTIAL", "key1"]
        assert parser.get_keys(connection, *args) == ["key1"]
        connection.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "module.cmd", "partial", "key1"
        )

    def test_key_positions_are_reported(self):
        args = ("XREAD", "COUNT", 2, "STREAMS", "mystream", "writers", 0, 0)
        specs = movablekeys_commands_table()["xread"]["key_specifications"]
        assert _get_keys_from_key_specs(args, specs) == [
            (4, "mystream"),
            (5, "writers"),
        ]

    def test_resp2_flat_key_spec_shape(self):
        # RESP2 reports every key spec section as a flat [name, value, ...]
        # sequence rather than a map; the interpretation is the same.
        spec = [
            b"flags",
            [b"RO", b"access"],
            b"begin_search",
            [b"type", b"keyword", b"spec", [b"keyword", b"STREAMS", b"startfrom", -1]],
            b"find_keys",
            [b"type", b"range", b"spec", [b"lastkey", -1, b"keystep", 1, b"limit", 2]],
        ]
        args = ("XREAD", "STREAMS", "mystream", "writers", 0, 0)
        assert _get_keys_from_key_specs(args, [spec]) == [
            (2, "mystream"),
            (3, "writers"),
        ]
