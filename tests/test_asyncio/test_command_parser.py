import pytest

from redis._parsers import AsyncCommandsParser
from redis._parsers.commands import _interpret_key_specs
from redis.commands.metadata import RequestPolicy, ResponsePolicy
from tests.conftest import skip_if_server_version_gte, skip_if_server_version_lt
from tests.helpers import get_expected_command_policies


def key_spec(flags, begin_search, find_keys):
    """A key specification in the RESP3 (map) shape of a ``COMMAND`` reply."""
    return {"flags": flags, "begin_search": begin_search, "find_keys": find_keys}


def movablekeys_command(name, key_specifications):
    command = {
        "name": name,
        "arity": -2,
        "flags": ["readonly", "movablekeys"],
        "first_key_pos": 0,
        "last_key_pos": 0,
        "step_count": 0,
    }
    if key_specifications is not None:
        command["key_specifications"] = key_specifications
    return command


XREAD_COMMAND = movablekeys_command(
    "xread",
    [
        key_spec(
            ["RO", "access"],
            {"keyword": {"keyword": "STREAMS", "startfrom": 1}},
            {"range": {"lastkey": -1, "keystep": 1, "limit": 2}},
        )
    ],
)

ZUNION_COMMAND = movablekeys_command(
    "zunion",
    [
        key_spec(
            ["RO", "access"],
            {"index": {"pos": 1}},
            {"keynum": {"keynumidx": 0, "firstkey": 1, "keystep": 1}},
        )
    ],
)

ZUNIONSTORE_COMMAND = movablekeys_command(
    "zunionstore",
    [
        key_spec(
            ["OW", "update"],
            {"index": {"pos": 1}},
            {"range": {"lastkey": 0, "keystep": 1, "limit": 0}},
        ),
        key_spec(
            ["RO", "access"],
            {"index": {"pos": 2}},
            {"keynum": {"keynumidx": 0, "firstkey": 1, "keystep": 1}},
        ),
    ],
)

# A module command whose keys live behind a movablekeys subcommand, shaped the
# way ``COMMAND`` reports nested subcommand entries.
MODULE_COMMAND = {
    "name": "modulecmd",
    "arity": -2,
    "flags": ["readonly", "module"],
    "first_key_pos": 0,
    "last_key_pos": 0,
    "step_count": 0,
    "subcommands": [
        [
            "modulecmd|get",
            -3,
            ["readonly", "movablekeys"],
            0,
            0,
            0,
            ["@read"],
            [],
            [
                key_spec(
                    ["RO", "access"],
                    {"keyword": {"keyword": "KEYS", "startfrom": 2}},
                    {"range": {"lastkey": -1, "keystep": 1, "limit": 0}},
                )
            ],
            [],
        ]
    ],
}


class FakeCommandsNode:
    """
    Stands in for the cluster node an ``AsyncCommandsParser`` resolves keys
    against.

    ``execute_command`` serves only ``COMMAND GETKEYS`` and records the call,
    so a test can assert the fallback ran - or never did.
    """

    def __init__(self, getkeys_result=None):
        self._getkeys_result = getkeys_result
        self.getkeys_calls = []

    async def execute_command(self, *args):
        assert args[0] == "COMMAND GETKEYS"
        self.getkeys_calls.append(args[1:])
        if self._getkeys_result is None:
            raise AssertionError("COMMAND GETKEYS should not have been called")
        return self._getkeys_result


class TestAsyncKeySpecInterpretation:
    """
    Local interpretation of ``COMMAND`` key specifications for ``movablekeys``
    commands: the parser resolves the keys without a ``COMMAND GETKEYS``
    roundtrip whenever the specifications fully describe the key positions,
    and falls back to the server only when they do not.
    """

    def make_parser(self, *commands, getkeys_result=None):
        parser = AsyncCommandsParser()
        parser.commands = {command["name"]: command for command in commands}
        parser.node = FakeCommandsNode(getkeys_result)
        return parser, parser.node

    @pytest.mark.asyncio
    async def test_begin_search_keyword_and_find_keys_range(self):
        parser, node = self.make_parser(XREAD_COMMAND)
        args = ["XREAD", "COUNT", 2, "STREAMS", "mystream", "writers", 0, 0]
        assert await parser.get_keys(*args) == ["mystream", "writers"]
        assert node.getkeys_calls == []

    @pytest.mark.asyncio
    async def test_keyword_matches_case_insensitively_and_keys_stay_raw(self):
        parser, node = self.make_parser(XREAD_COMMAND)
        args = ["XREAD", b"streams", b"mystream", 0]
        assert await parser.get_keys(*args) == [b"mystream"]
        assert node.getkeys_calls == []

    @pytest.mark.asyncio
    async def test_find_keys_keynum(self):
        parser, node = self.make_parser(ZUNION_COMMAND)
        args = ["ZUNION", 2, "zset1", "zset2"]
        assert await parser.get_keys(*args) == ["zset1", "zset2"]
        # The numkeys argument may arrive as bytes.
        assert await parser.get_keys("ZUNION", b"1", b"zset1") == [b"zset1"]
        assert node.getkeys_calls == []

    @pytest.mark.asyncio
    async def test_find_keys_keynum_zero_keys(self):
        parser, node = self.make_parser(ZUNION_COMMAND)
        assert await parser.get_keys("ZUNION", 0) == []
        assert node.getkeys_calls == []

    @pytest.mark.asyncio
    async def test_multiple_key_specifications(self):
        parser, node = self.make_parser(ZUNIONSTORE_COMMAND)
        args = ["ZUNIONSTORE", "out", 2, "zset1", "zset2", "WEIGHTS", 2, 3]
        assert await parser.get_keys(*args) == ["out", "zset1", "zset2"]
        assert node.getkeys_calls == []

    @pytest.mark.asyncio
    async def test_movablekeys_module_subcommand(self):
        parser, node = self.make_parser(MODULE_COMMAND)
        args = ["MODULECMD", "GET", "config", "KEYS", "key1", "key2"]
        assert await parser.get_keys(*args) == ["key1", "key2"]
        assert node.getkeys_calls == []

    @pytest.mark.asyncio
    async def test_resp2_flat_key_specification_shape(self):
        resp2_xread = movablekeys_command(
            "xread",
            [
                [
                    b"flags",
                    [b"RO", b"access"],
                    b"begin_search",
                    [b"keyword", [b"keyword", b"STREAMS", b"startfrom", 1]],
                    b"find_keys",
                    [b"range", [b"lastkey", -1, b"keystep", 1, b"limit", 2]],
                ]
            ],
        )
        parser, node = self.make_parser(resp2_xread)
        args = ["XREAD", "STREAMS", "mystream", 0]
        assert await parser.get_keys(*args) == ["mystream"]
        assert node.getkeys_calls == []

    @pytest.mark.asyncio
    async def test_negative_lastkey_and_keystep(self):
        command = movablekeys_command(
            "cmd",
            [
                key_spec(
                    ["RO"],
                    {"index": {"pos": 1}},
                    {"range": {"lastkey": -2, "keystep": 2, "limit": 0}},
                )
            ],
        )
        parser, node = self.make_parser(command)
        args = ["CMD", "key1", "value1", "key2", "value2", "stop"]
        assert await parser.get_keys(*args) == ["key1", "key2"]
        assert node.getkeys_calls == []

    @pytest.mark.asyncio
    async def test_negative_keyword_startfrom(self):
        command = movablekeys_command(
            "cmd",
            [
                key_spec(
                    ["RO"],
                    {"keyword": {"keyword": "KEYS", "startfrom": -4}},
                    {"range": {"lastkey": -1, "keystep": 1, "limit": 0}},
                )
            ],
        )
        parser, node = self.make_parser(command)
        args = ["CMD", "arg", "KEYS", "key1"]
        assert await parser.get_keys(*args) == ["key1"]
        assert node.getkeys_calls == []

    @pytest.mark.asyncio
    async def test_not_key_specification_contributes_no_keys(self):
        command = movablekeys_command(
            "cmd",
            [
                key_spec(
                    ["RO", "not_key"],
                    {"index": {"pos": 1}},
                    {"range": {"lastkey": 0, "keystep": 1, "limit": 0}},
                )
            ],
        )
        parser, node = self.make_parser(command)
        # The covered argument is not a key, so it must not be routed by.
        assert await parser.get_keys("CMD", "not-a-key") == []
        assert node.getkeys_calls == []

    @pytest.mark.asyncio
    async def test_incomplete_specification_falls_back_to_getkeys(self):
        command = movablekeys_command(
            "georadius",
            [
                key_spec(
                    ["RW", "access"],
                    {"index": {"pos": 1}},
                    {"range": {"lastkey": 0, "keystep": 1, "limit": 0}},
                ),
                key_spec(
                    ["OW", "update", "incomplete"],
                    {"keyword": {"keyword": "STORE", "startfrom": 6}},
                    {"range": {"lastkey": 0, "keystep": 1, "limit": 0}},
                ),
            ],
        )
        parser, node = self.make_parser(command, getkeys_result=[b"Sicily", b"out"])
        args = ["GEORADIUS", "Sicily", 15, 37, 200, "km", "STORE", "out"]
        assert await parser.get_keys(*args) == [b"Sicily", b"out"]
        assert node.getkeys_calls == [tuple(args)]

    @pytest.mark.asyncio
    async def test_missing_keyword_falls_back_to_getkeys(self):
        parser, node = self.make_parser(XREAD_COMMAND, getkeys_result=[])
        assert await parser.get_keys("XREAD", "COUNT", 2) == []
        assert node.getkeys_calls == [("XREAD", "COUNT", 2)]

    @pytest.mark.asyncio
    async def test_unknown_find_keys_type_falls_back_to_getkeys(self):
        command = movablekeys_command(
            "cmd", [key_spec(["RO"], {"index": {"pos": 1}}, {"mystery": {}})]
        )
        parser, node = self.make_parser(command, getkeys_result=[b"key1"])
        assert await parser.get_keys("CMD", "key1") == [b"key1"]
        assert node.getkeys_calls == [("CMD", "key1")]

    @pytest.mark.asyncio
    async def test_missing_key_specifications_fall_back_to_getkeys(self):
        parser, node = self.make_parser(
            movablekeys_command("cmd", None), getkeys_result=[b"key1"]
        )
        assert await parser.get_keys("CMD", "key1") == [b"key1"]
        assert node.getkeys_calls == [("CMD", "key1")]

    def test_keys_are_returned_with_their_positions(self):
        args = ("XREAD", "COUNT", 2, "STREAMS", "mystream", "writers", 0, 0)
        keys = _interpret_key_specs(args, XREAD_COMMAND["key_specifications"])
        assert keys == [(4, "mystream"), (5, "writers")]


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
