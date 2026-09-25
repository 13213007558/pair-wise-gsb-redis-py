from unittest.mock import AsyncMock, Mock

import pytest

from redis._parsers import AsyncCommandsParser, CommandsParser
from redis.commands.metadata import RequestPolicy, ResponsePolicy
from tests.conftest import skip_if_server_version_gte, skip_if_server_version_lt
from tests.helpers import get_expected_command_policies, movablekeys_commands_table


@pytest.mark.onlycluster
@skip_if_server_version_lt("8.0.0")
class TestAsyncCommandParser:
    @pytest.mark.asyncio
    @skip_if_server_version_gte("8.5.240")
    async def test_get_command_policies(self, r):
        commands_parser = AsyncCommandsParser()
        await commands_parser.initialize(node=r.get_default_node())
        expected_command_policies = get_expected_command_policies()

        actual_policies = await commands_parser.get_command_policies()
        assert len(actual_policies) > 0

        for module_name, commands in expected_command_policies.items():
            for command, command_policies in commands.items():
                assert command in actual_policies[module_name]
                assert command_policies == [
                    command,
                    actual_policies[module_name][command].request_policy,
                    actual_policies[module_name][command].response_policy,
                ]

    @skip_if_server_version_lt("7.0.0")
    @pytest.mark.asyncio
    async def test_get_movablekeys_from_key_specs(self, r):
        """
        movablekeys commands with complete key specifications are interpreted
        locally, so the keys are the invocation's own arguments and no
        COMMAND GETKEYS round trip is needed.
        """
        commands_parser = AsyncCommandsParser()
        await commands_parser.initialize(node=r.get_default_node())

        args1 = ["XREAD", "COUNT", 2, b"STREAMS", "mystream", "writers", 0, 0]
        args2 = ["ZUNIONSTORE", "out", 2, "zset1", "zset2", "WEIGHTS", 2, 3]
        args3 = ["GEORADIUS", "Sicily", 15, 37, 200, "km", "WITHCOORD", b"STORE", "out"]

        assert sorted(await commands_parser.get_keys(*args1)) == ["mystream", "writers"]
        assert sorted(await commands_parser.get_keys(*args2)) == [
            "out",
            "zset1",
            "zset2",
        ]
        assert sorted(await commands_parser.get_keys(*args3)) == ["Sicily", "out"]

    @skip_if_server_version_lt("8.5.240")
    @pytest.mark.asyncio
    async def test_get_command_policies_json_debug_updated(self, r):
        commands_parser = AsyncCommandsParser()
        await commands_parser.initialize(node=r.get_default_node())
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

        actual_policies = await commands_parser.get_command_policies()
        assert len(actual_policies) > 0

        for module_name, commands in expected_command_policies.items():
            for command, command_policies in commands.items():
                assert command in actual_policies[module_name]
                assert command_policies == [
                    command,
                    actual_policies[module_name][command].request_policy,
                    actual_policies[module_name][command].response_policy,
                ]


class TestAsyncKeySpecInterpretation:
    """
    Local interpretation of `movablekeys` commands' key specifications.

    The parser is built over a synthetic `COMMAND` reply, so no server is
    needed; a mock node stands in for the `COMMAND GETKEYS` fallback and
    records whether it was consulted.
    """

    @staticmethod
    def parser(getkeys_return=None):
        parser = AsyncCommandsParser()
        parser.commands = movablekeys_commands_table()
        parser.node = Mock()
        parser.node.execute_command = AsyncMock(return_value=getkeys_return or [])
        return parser

    @pytest.mark.asyncio
    async def test_xread_keyword_begin_search_and_limit(self):
        # The keys after STREAMS are half of the remaining arguments; the IDs
        # and the COUNT option must not be mistaken for keys.
        parser = self.parser()
        args = ["XREAD", "COUNT", 2, "STREAMS", "mystream", "writers", 0, 0]
        assert await parser.get_keys(*args) == ["mystream", "writers"]
        parser.node.execute_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_zunionstore_multiple_specs(self):
        parser = self.parser()
        args = ["ZUNIONSTORE", "out", 2, "zset1", "zset2", "WEIGHTS", 2, 3]
        assert await parser.get_keys(*args) == ["out", "zset1", "zset2"]
        parser.node.execute_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_keynum_zero_keys(self):
        parser = self.parser()
        args = ["EVAL", "return {ARGV[1]}", 0, "key1"]
        assert await parser.get_keys(*args) == []

    @pytest.mark.asyncio
    async def test_not_key_spec_contributes_no_keys(self):
        parser = self.parser()
        assert await parser.get_keys("SHARD.PUB", "channel") == []
        parser.node.execute_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_incomplete_spec_falls_back_to_getkeys(self):
        parser = self.parser(getkeys_return=["key1"])
        assert await parser.get_keys("INCOMPLETE.CMD", "key1") == ["key1"]
        parser.node.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "INCOMPLETE.CMD", "key1"
        )

    @pytest.mark.asyncio
    async def test_unknown_find_keys_falls_back_to_getkeys(self):
        parser = self.parser(getkeys_return=["key1"])
        assert await parser.get_keys("UNKNOWNFIND.CMD", "key1") == ["key1"]
        parser.node.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "UNKNOWNFIND.CMD", "key1"
        )

    @pytest.mark.asyncio
    async def test_movablekeys_subcommand_of_module_command(self):
        parser = self.parser()
        args = ["MODULE.CMD SUB", "key1", "key2"]
        assert await parser.get_keys(*args) == ["key1", "key2"]
        parser.node.execute_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_movablekeys_subcommand_falls_back_to_getkeys(self):
        parser = self.parser(getkeys_return=["key1"])
        args = ["MODULE.CMD PARTIAL", "key1"]
        assert await parser.get_keys(*args) == ["key1"]
        parser.node.execute_command.assert_called_once_with(
            "COMMAND GETKEYS", "module.cmd", "partial", "key1"
        )

    @pytest.mark.asyncio
    async def test_sync_and_async_interpretations_agree(self):
        sync_connection = Mock()
        sync_connection.command.return_value = {}
        sync_parser = CommandsParser(sync_connection)
        async_parser = self.parser()
        commands = movablekeys_commands_table()
        sync_parser.commands = commands
        async_parser.commands = commands

        invocations = [
            ("XREAD", "COUNT", 2, "STREAMS", "mystream", "writers", 0, 0),
            ("XREAD", b"COUNT", 2, b"streams", b"mystream", 0),
            ("XREAD", "STREAMS"),
            ("ZUNION", 2, "zset1", "zset2", "WEIGHTS", 2, 3),
            ("ZUNIONSTORE", "out", 2, "zset1", "zset2", "WEIGHTS", 2, 3),
            ("ZUNIONSTORE", "out", 0),
            ("GEORADIUS", "Sicily", 15, 37, 200, "km", "WITHCOORD", "STORE", "out"),
            ("GEORADIUS", "Sicily", 15, 37, 200, "km"),
            ("EVAL", "return 1", 2, "key1", "key2", "arg"),
            ("EVAL", "return 1", 0, "key1"),
            ("MODULE.CMD SUB", "key1", "key2"),
            ("SHARD.PUB", "channel"),
            ("KEYSTEP.CMD", "key1", "value1", "key2", "value2"),
        ]
        for args in invocations:
            sync_keys = sync_parser.get_keys(sync_connection, *args)
            async_keys = await async_parser.get_keys(*args)
            assert sync_keys == async_keys, args
