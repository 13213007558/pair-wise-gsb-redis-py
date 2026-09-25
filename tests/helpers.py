import logging
from time import sleep
from typing import Callable

from redis.commands.metadata import RequestPolicy, ResponsePolicy


def wait_for_condition(
    predicate: Callable[[], bool],
    timeout: float = 0.2,
    check_interval: float = 0.01,
    error_message: str = "Timeout waiting for condition",
) -> None:
    """
    Poll a condition until it becomes True or timeout is reached.

    Args:
        predicate: A callable that returns True when the condition is met
        timeout: Maximum time to wait in seconds (default: 0.2s = 20 * 0.01s)
        check_interval: Time to sleep between checks in seconds (default: 0.01s)
        error_message: Error message to raise if timeout occurs

    Raises:
        AssertionError: If the condition is not met within the timeout period

    Example:
        # Wait for circuit breaker to open
        wait_for_condition(
            lambda: cb2.state == CBState.OPEN,
            timeout=0.2,
            error_message="Timeout waiting for cb2 to open"
        )

        # Wait for failover strategy to select a specific database
        wait_for_condition(
            lambda: client.command_executor.active_database is mock_db,
            timeout=0.2,
            error_message="Timeout waiting for active database to change"
        )
    """
    max_retries = int(timeout / check_interval)

    for attempt in range(max_retries):
        if predicate():
            logging.debug(f"Condition met after {attempt} attempts")
            return
        sleep(check_interval)

    raise AssertionError(error_message)


def get_expected_command_policies(changes_in_defaults={}):
    default_cmd_policies = {
        "core": {
            "keys": [
                "keys",
                RequestPolicy.ALL_SHARDS,
                ResponsePolicy.DEFAULT_KEYLESS,
            ],
            "acl setuser": [
                "acl setuser",
                RequestPolicy.ALL_NODES,
                ResponsePolicy.ALL_SUCCEEDED,
            ],
            "exists": ["exists", RequestPolicy.MULTI_SHARD, ResponsePolicy.AGG_SUM],
            "config resetstat": [
                "config resetstat",
                RequestPolicy.ALL_NODES,
                ResponsePolicy.ALL_SUCCEEDED,
            ],
            "slowlog len": [
                "slowlog len",
                RequestPolicy.ALL_NODES,
                ResponsePolicy.AGG_SUM,
            ],
            "scan": ["scan", RequestPolicy.SPECIAL, ResponsePolicy.SPECIAL],
            "latency history": [
                "latency history",
                RequestPolicy.ALL_NODES,
                ResponsePolicy.SPECIAL,
            ],
            "memory doctor": [
                "memory doctor",
                RequestPolicy.ALL_SHARDS,
                ResponsePolicy.SPECIAL,
            ],
            "randomkey": [
                "randomkey",
                RequestPolicy.ALL_SHARDS,
                ResponsePolicy.SPECIAL,
            ],
            "mget": [
                "mget",
                RequestPolicy.MULTI_SHARD,
                ResponsePolicy.DEFAULT_KEYED,
            ],
            "function restore": [
                "function restore",
                RequestPolicy.ALL_SHARDS,
                ResponsePolicy.ALL_SUCCEEDED,
            ],
        },
        "json": {
            "debug": [
                "debug",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
            "get": [
                "get",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
        },
        "ft": {
            "search": [
                "search",
                RequestPolicy.DEFAULT_KEYLESS,
                ResponsePolicy.DEFAULT_KEYLESS,
            ],
            "create": [
                "create",
                RequestPolicy.DEFAULT_KEYLESS,
                ResponsePolicy.DEFAULT_KEYLESS,
            ],
        },
        "bf": {
            "add": [
                "add",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
            "madd": [
                "madd",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
        },
        "cf": {
            "add": [
                "add",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
            "mexists": [
                "mexists",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
        },
        "tdigest": {
            "add": [
                "add",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
            "min": [
                "min",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
        },
        "ts": {
            "create": [
                "create",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
            "info": [
                "info",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
        },
        "topk": {
            "list": [
                "list",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
            "query": [
                "query",
                RequestPolicy.DEFAULT_KEYED,
                ResponsePolicy.DEFAULT_KEYED,
            ],
        },
    }
    default_cmd_policies.update(changes_in_defaults)
    return default_cmd_policies


def key_spec(flags, begin_search, find_keys):
    """A single key specification, in the RESP3 (map) shape of a `COMMAND` reply."""
    return {"flags": flags, "begin_search": begin_search, "find_keys": find_keys}


def begin_index(pos):
    return {"type": "index", "spec": {"pos": pos}}


def begin_keyword(keyword, startfrom):
    return {"type": "keyword", "spec": {"keyword": keyword, "startfrom": startfrom}}


def find_range(lastkey, keystep=1, limit=0):
    return {
        "type": "range",
        "spec": {"lastkey": lastkey, "keystep": keystep, "limit": limit},
    }


def find_keynum(keynumidx=0, firstkey=1, keystep=1):
    return {
        "type": "keynum",
        "spec": {"keynumidx": keynumidx, "firstkey": firstkey, "keystep": keystep},
    }


def movablekeys_command_entry(key_specifications, subcommands=None):
    """
    A parsed `COMMAND` reply entry for a `movablekeys` command: the legacy key
    positions are zeroed, so the keys are only discoverable through the specs.
    """
    return {
        "name": "",
        "arity": -2,
        "flags": ["readonly", "movablekeys"],
        "first_key_pos": 0,
        "last_key_pos": 0,
        "step_count": 0,
        "tips": [],
        "key_specifications": key_specifications,
        "subcommands": subcommands or [],
    }


def movablekeys_commands_table():
    """
    A synthetic parsed `COMMAND` reply of `movablekeys` commands, in the RESP3
    (map) key-spec shape, for exercising local key-spec interpretation without a
    server. Both command parser test suites build their parsers from this one
    table, so the sync and async interpretations are compared over identical
    metadata.
    """
    return {
        # XREAD [COUNT n] [BLOCK ms] STREAMS key [key ...] id [id ...]: the keys
        # after STREAMS are half of the remaining arguments (limit 2).
        "xread": movablekeys_command_entry(
            [
                key_spec(
                    ["RO", "access"],
                    begin_keyword("STREAMS", -1),
                    find_range(-1, limit=2),
                )
            ]
        ),
        # ZUNION numkeys key [key ...] ...: the key count is read off the invocation.
        "zunion": movablekeys_command_entry(
            [key_spec(["RO", "access"], begin_index(1), find_keynum())]
        ),
        # ZUNIONSTORE destination numkeys key [key ...] ...: two specs, one for the
        # destination and one for the source sets.
        "zunionstore": movablekeys_command_entry(
            [
                key_spec(["RW", "insert"], begin_index(1), find_range(0)),
                key_spec(["RO", "access"], begin_index(2), find_keynum()),
            ]
        ),
        # GEORADIUS key ... [STORE key] [STOREDIST key]: two optional keyword
        # specs, found by scanning forward from the first argument.
        "georadius": movablekeys_command_entry(
            [
                key_spec(["RO", "access"], begin_index(1), find_range(0)),
                key_spec(["RW", "insert"], begin_keyword("STORE", 1), find_range(0)),
                key_spec(
                    ["RW", "insert"], begin_keyword("STOREDIST", 1), find_range(0)
                ),
            ]
        ),
        # EVAL script numkeys key [key ...] arg [arg ...]
        "eval": movablekeys_command_entry(
            [key_spec(["RW", "access"], begin_index(2), find_keynum())]
        ),
        # A container module command with movablekeys subcommands.
        "module.cmd": {
            "name": "module.cmd",
            "arity": -2,
            "flags": ["container"],
            "first_key_pos": 0,
            "last_key_pos": 0,
            "step_count": 0,
            "tips": [],
            "key_specifications": [],
            "subcommands": [
                [
                    "module.cmd|sub",
                    -3,
                    ["readonly", "movablekeys"],
                    0,
                    0,
                    0,
                    [],
                    [],
                    [key_spec(["RO", "access"], begin_index(2), find_range(-1))],
                    [],
                ],
                [
                    "module.cmd|partial",
                    -3,
                    ["readonly", "movablekeys"],
                    0,
                    0,
                    0,
                    [],
                    [],
                    [
                        key_spec(
                            ["RO", "access", "incomplete"],
                            begin_index(2),
                            find_range(-1),
                        )
                    ],
                    [],
                ],
            ],
        },
        # A command whose only spec describes an argument that is not a key name.
        "shard.pub": movablekeys_command_entry(
            [key_spec(["RO", "not_key"], begin_index(1), find_range(-1))]
        ),
        # Specs that cannot be interpreted locally in full: each of these must be
        # resolved by COMMAND GETKEYS instead.
        "incomplete.cmd": movablekeys_command_entry(
            [key_spec(["RW", "access", "incomplete"], begin_index(1), find_range(-1))]
        ),
        "varflags.cmd": movablekeys_command_entry(
            [
                key_spec(
                    ["RW", "access", "variable_flags"], begin_index(1), find_range(-1)
                )
            ]
        ),
        "unknownbegin.cmd": movablekeys_command_entry(
            [key_spec(["RO"], {"type": "unknown", "spec": {}}, find_range(-1))]
        ),
        "unknownfind.cmd": movablekeys_command_entry(
            [key_spec(["RO"], begin_index(1), {"type": "unknown", "spec": {}})]
        ),
        # A range with a step: keys and values alternate.
        "keystep.cmd": movablekeys_command_entry(
            [key_spec(["RO"], begin_index(1), find_range(-1, keystep=2))]
        ),
    }
