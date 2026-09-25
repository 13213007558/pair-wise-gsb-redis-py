from unittest.mock import AsyncMock, Mock

import pytest

from redis._parsers import AsyncCommandsParser
from redis._parsers.commands import _extract_keys_from_specs
from redis.commands.metadata import RequestPolicy, ResponsePolicy
from redis.exceptions import ResponseError
from tests.conftest import skip_if_server_version_gte, skip_if_server_version_lt
from tests.helpers import get_expected_command_policies
from tests.test_command_parser import (
    RESP_SHAPES,
    TAGGED_SLOT,
    _command_entry,
    _index_spec,
    _keynum_spec,
    _keyword_spec,
    _slot,
    _subcommand_entry,
)


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


def _parser_for(commands):
    """An AsyncCommandsParser with scripted COMMAND metadata and no network."""
    parser = AsyncCommandsParser()
    parser.commands = commands
    parser.node = Mock()
    parser.node.execute_command = AsyncMock(
        side_effect=AssertionError("key extraction issued a server round trip")
    )
    return parser


class TestAsyncLocalKeyExtraction:
    """
    The async mirror of tests.test_command_parser.TestLocalKeyExtraction: key
    extraction from Redis 7 key specifications shares its pure logic with the
    sync parser, so the same scripted COMMAND metadata must yield the same keys
    without a COMMAND GETKEYS round trip.
    """

    @RESP_SHAPES
    async def test_index_and_range_specs(self, shape):
        parser = _parser_for(
            {
                "bitop": _command_entry(
                    "bitop",
                    ["write", "movablekeys"],
                    [
                        shape(_index_spec(["RW", "update"], 2)),
                        shape(_index_spec(["RO", "access"], 3, lastkey=-1)),
                    ],
                )
            }
        )
        args = ["BITOP", "AND", "{k}dest", "{k}1", "{k}2", "{k}3"]

        keys = await parser.get_keys(*args)

        assert keys == ["{k}dest", "{k}1", "{k}2", "{k}3"]
        assert {_slot(key) for key in keys} == {TAGGED_SLOT}

    @RESP_SHAPES
    async def test_keyword_spec_and_optional_argument(self, shape):
        parser = _parser_for(
            {
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
        )

        args = ["GEORADIUS", "Sicily", 15, 37, 200, "km", "WITHCOORD"]
        assert await parser.get_keys(*args) == ["Sicily"]

        for store in ["STORE", "store", b"STORE", b"Store"]:
            args = ["GEORADIUS", "Sicily", 15, 37, 200, "km", store, "out"]
            keys = await parser.get_keys(*args)
            assert keys == ["Sicily", "out"]
            assert [_slot(key) for key in keys] == [10713, 16101]

    @RESP_SHAPES
    async def test_keyword_search_from_the_end_with_limit(self, shape):
        parser = _parser_for(
            {
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
        )

        args = ["XREAD", "COUNT", 2, "STREAMS", "mystream", "writers", 0, 0]
        keys = await parser.get_keys(*args)
        assert keys == ["mystream", "writers"]
        assert [_slot(key) for key in keys] == [12571, 14985]

    @RESP_SHAPES
    async def test_keynum_spec(self, shape):
        parser = _parser_for(
            {
                "eval": _command_entry(
                    "eval",
                    ["movablekeys"],
                    [shape(_keynum_spec(["RW", "access", "update"], 2))],
                )
            }
        )

        for numkeys in [2, "2", b"2"]:
            args = ["EVAL", "return 1", numkeys, "{k}1", "{k}2", "arg"]
            keys = await parser.get_keys(*args)
            assert keys == ["{k}1", "{k}2"]
            assert {_slot(key) for key in keys} == {TAGGED_SLOT}

        assert await parser.get_keys(*["EVAL", "return 1", 0]) == []

    @RESP_SHAPES
    async def test_invalid_keynum_raises_before_sending(self, shape):
        parser = _parser_for(
            {
                "eval": _command_entry(
                    "eval",
                    ["movablekeys"],
                    [shape(_keynum_spec(["RW", "access", "update"], 2))],
                )
            }
        )

        # The malformed invocation is reported locally: the parser's node raises
        # AssertionError if a command is sent over it.
        with pytest.raises(ResponseError, match="Invalid arguments"):
            await parser.get_keys(*["EVAL", "return 1", "not-a-number", "{k}1"])

    @RESP_SHAPES
    async def test_not_key_arguments_do_not_decide_the_slot(self, shape):
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

        assert _extract_keys_from_specs(commands["cmd"], args) == [
            ("{k}1", {"RO", "access"}),
            ("not-a-key", {"RO", "access", "not_key"}),
        ]
        keys = await parser.get_keys(*args)
        assert keys == ["{k}1"]
        assert _slot(keys[0]) == TAGGED_SLOT

    @RESP_SHAPES
    async def test_incomplete_spec_falls_back_to_server(self, shape):
        parser = _parser_for(
            {
                "migrate": _command_entry(
                    "migrate",
                    ["write", "movablekeys"],
                    [
                        shape(_index_spec(["RW", "access", "incomplete"], 3)),
                        shape(_keyword_spec(["RW", "access"], "KEYS", -1, lastkey=-1)),
                    ],
                )
            }
        )
        parser.node.execute_command = AsyncMock(return_value=[b"key1", b"key2"])

        args = ["MIGRATE", "host", 6379, "", 0, 5000, "KEYS", "key1", "key2"]
        keys = await parser.get_keys(*args)

        assert keys == [b"key1", b"key2"]
        parser.node.execute_command.assert_called_once_with("COMMAND GETKEYS", *args)

    @RESP_SHAPES
    async def test_subcommand_key_extraction(self, shape):
        parser = _parser_for(
            {
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
        )

        for args in [
            ["MYCONTAINER", "GETALL", "{k}1"],
            ["mycontainer getall", "{k}1"],
        ]:
            keys = await parser.get_keys(*args)
            assert keys == ["{k}1"]
            assert _slot(keys[0]) == TAGGED_SLOT
