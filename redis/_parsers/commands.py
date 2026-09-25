from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from redis.commands.metadata import (
    CommandMetadata,
    CommandMetadataRecordsCache,
    CommandPolicies,
    PolicyRecords,
    RequestPolicy,
    ResponsePolicy,
)
from redis.exceptions import IncorrectPolicyType, RedisError, ResponseError
from redis.utils import deprecated_function, str_if_bytes

if TYPE_CHECKING:
    from redis.asyncio.cluster import ClusterNode

# The record type ``_build_command_records`` builds per command: policies or full metadata.
# Constrained to the two, rather than left open, so a ``to_record`` that builds anything else
# is a type error at the call site.
_RecordT = TypeVar("_RecordT", CommandPolicies, CommandMetadata)

# Re-exported for backwards compatibility: these types used to be defined here and are
# now owned by ``redis.commands.metadata``. Import them from their new home instead.
__all__ = [
    "AbstractCommandsParser",
    "AsyncCommandsParser",
    "CommandPolicies",
    "CommandsParser",
    "PolicyRecords",
    "RequestPolicy",
    "ResponsePolicy",
]


class AbstractCommandsParser:
    def _get_pubsub_keys(self, *args):
        """
        Get the keys from pubsub command.
        Although PubSub commands have predetermined key locations, they are not
        supported in the 'COMMAND's output, so the key positions are hardcoded
        in this method
        """
        if len(args) < 2:
            # The command has no keys in it
            return None
        args = [str_if_bytes(arg) for arg in args]
        command = args[0].upper()
        keys = None
        if command == "PUBSUB":
            # the second argument is a part of the command name, e.g.
            # ['PUBSUB', 'NUMSUB', 'foo'].
            pubsub_type = args[1].upper()
            if pubsub_type in ["CHANNELS", "NUMSUB", "SHARDCHANNELS", "SHARDNUMSUB"]:
                keys = args[2:]
        elif command in ["SUBSCRIBE", "PSUBSCRIBE", "UNSUBSCRIBE", "PUNSUBSCRIBE"]:
            # format example:
            # SUBSCRIBE channel [channel ...]
            keys = list(args[1:])
        elif command in ["PUBLISH", "SPUBLISH"]:
            # format example:
            # PUBLISH channel message
            keys = [args[1]]
        return keys

    def parse_subcommand(self, command, **options):
        return _parse_subcommand(command)


class CommandsParser(AbstractCommandsParser):
    """
    Parses Redis commands to get command keys.
    COMMAND output is used to determine key locations.
    Commands that do not have a predefined key location are flagged with
    'movablekeys', and these commands' keys are extracted locally from their
    key specifications when the server reports them (Redis 7.0+); commands
    whose key specifications are incomplete or missing still have their keys
    determined by the command 'COMMAND GETKEYS'.
    """

    def __init__(self, redis_connection):
        self.commands = {}
        self.redis_connection = redis_connection
        self.initialize(self.redis_connection)

    def initialize(self, r):
        commands = r.command()
        uppercase_commands = []
        for cmd in commands:
            if any(x.isupper() for x in cmd):
                uppercase_commands.append(cmd)
        for cmd in uppercase_commands:
            commands[cmd.lower()] = commands.pop(cmd)
        self.commands = commands

    # As soon as this PR is merged into Redis, we should reimplement
    # our logic to use COMMAND INFO changes to determine the key positions
    # https://github.com/redis/redis/pull/8324
    def get_keys(self, redis_conn, *args):
        """
        Get the keys from the passed command.

        Keys of 'movablekeys' commands are extracted locally from the command's
        key specifications when they are complete, falling back to
        'COMMAND GETKEYS' otherwise.

        NOTE: Due to a bug in redis<7.0, this function does not work properly
        for EVAL or EVALSHA when the `numkeys` arg is 0.
         - issue: https://github.com/redis/redis/issues/9493
         - fix: https://github.com/redis/redis/pull/9733

        So, don't use this function with EVAL or EVALSHA.
        """
        if len(args) < 2:
            # The command has no keys in it
            return None

        cmd_name = args[0].lower()
        if cmd_name not in self.commands:
            # try to split the command name and to take only the main command,
            # e.g. 'memory' for 'memory usage'
            cmd_name_split = cmd_name.split()
            cmd_name = cmd_name_split[0]
            if cmd_name in self.commands:
                # save the split command to args
                args = cmd_name_split + list(args[1:])
            else:
                # We'll try to reinitialize the commands cache, if the engine
                # version has changed, the commands may not be current
                self.initialize(redis_conn)
                if cmd_name not in self.commands:
                    raise RedisError(
                        f"{cmd_name.upper()} command doesn't exist in Redis commands"
                    )

        command = self.commands.get(cmd_name)
        # A container subcommand carries its own flags and key specifications, so
        # resolve it before deciding how the keys are found.
        effective_command = command
        if len(args) > 1 and command.get("subcommands"):
            subcmd_name = f"{cmd_name}|{args[1].lower()}"
            for subcmd in command["subcommands"]:
                if str_if_bytes(subcmd[0]) == subcmd_name:
                    effective_command = self.parse_subcommand(subcmd)
                    break
        if "movablekeys" in effective_command["flags"]:
            extracted = _extract_keys_from_specs(effective_command, args)
            if extracted is not None:
                # Arguments flagged not_key (e.g. a shard pubsub channel) are not
                # key names and must not decide the slot.
                keys = [key for key, flags in extracted if "not_key" not in flags]
            else:
                keys = self._get_moveable_keys(redis_conn, *args)
        elif "pubsub" in command["flags"] or command["name"] == "pubsub":
            keys = self._get_pubsub_keys(*args)
        else:
            if (
                command["step_count"] == 0
                and command["first_key_pos"] == 0
                and command["last_key_pos"] == 0
            ):
                is_subcmd = False
                if "subcommands" in command:
                    subcmd_name = f"{cmd_name}|{args[1].lower()}"
                    for subcmd in command["subcommands"]:
                        if str_if_bytes(subcmd[0]) == subcmd_name:
                            command = self.parse_subcommand(subcmd)

                            if command["first_key_pos"] > 0:
                                is_subcmd = True

                # The command doesn't have keys in it
                if not is_subcmd:
                    return None
            last_key_pos = command["last_key_pos"]
            if last_key_pos < 0:
                last_key_pos = len(args) - abs(last_key_pos)
            keys_pos = list(
                range(command["first_key_pos"], last_key_pos + 1, command["step_count"])
            )
            keys = [args[pos] for pos in keys_pos]

        return keys

    def _get_moveable_keys(self, redis_conn, *args):
        """
        NOTE: Due to a bug in redis<7.0, this function does not work properly
        for EVAL or EVALSHA when the `numkeys` arg is 0.
         - issue: https://github.com/redis/redis/issues/9493
         - fix: https://github.com/redis/redis/pull/9733

        So, don't use this function with EVAL or EVALSHA.
        """
        # The command name should be split into separate arguments,
        # e.g. 'MEMORY USAGE' will be split into ['MEMORY', 'USAGE']
        pieces = args[0].split() + list(args[1:])
        try:
            keys = redis_conn.execute_command("COMMAND GETKEYS", *pieces)
        except ResponseError as e:
            message = e.__str__()
            if (
                "Invalid arguments" in message
                or "The command has no key arguments" in message
            ):
                return None
            else:
                raise e
        return keys

    @deprecated_function(
        version="8.2.0",
        reason="Use get_commands_metadata_cache() instead.",
    )
    def get_command_policies(self) -> PolicyRecords:
        """
        Retrieve and process the command policies for all commands and subcommands.

        DEPRECATED: use :meth:`get_commands_metadata_cache` instead. Nothing in this library calls this
        method any more, and it will be removed in a future release. A metadata resolver keeps a
        compatibility shim for an object that serves only this method, because it was the whole
        contract in 7.1.0; that shim is not an invitation to write one. To decide routing
        yourself, implement the public ``PolicyResolver`` ABC instead.

        This method traverses through commands and subcommands, extracting policy details
        from associated data structures and constructing a dictionary of commands with their
        associated policies. It supports nested data structures and handles both main commands
        and their subcommands.

        This is the routing view of :meth:`get_commands_metadata_cache`: the same traversal of the
        same reply, keyed the same way, projected down to the two policies the cluster client
        routes by. A caller that also needs the client-side-caching metadata asks for the
        metadata records instead.

        Returns:
            PolicyRecords: A collection of commands and subcommands associated with their
            respective policies.

        Raises:
            IncorrectPolicyType: If an invalid policy type is encountered during policy extraction.
        """
        return _build_policy_records(self.commands)

    def get_commands_metadata_cache(self) -> CommandMetadataRecordsCache:
        """
        Retrieve and process the metadata records cache for all commands and subcommands.

        This method normalizes the command flags, command tips and key metadata of the
        ``COMMAND`` output into metadata records, keyed the same way as the policy records
        that the deprecated ``get_command_policies`` returns. The routing policies each
        record carries are the ones that method resolves, so both views of the same reply
        stay in step.

        Returns:
            CommandMetadataRecordsCache: A collection of commands and subcommands
            associated with their respective metadata.

        Raises:
            IncorrectPolicyType: If an invalid policy type is encountered during policy extraction.
        """
        return _build_commands_metadata_cache(self.commands)


class AsyncCommandsParser(AbstractCommandsParser):
    """
    Parses Redis commands to get command keys.

    COMMAND output is used to determine key locations.
    Commands that do not have a predefined key location are flagged with 'movablekeys',
    and these commands' keys are extracted locally from their key specifications when
    the server reports them (Redis 7.0+); commands whose key specifications are
    incomplete or missing still have their keys determined by 'COMMAND GETKEYS'.

    NOTE: Due to a bug in redis<7.0, this does not work properly
    for EVAL or EVALSHA when the `numkeys` arg is 0.
     - issue: https://github.com/redis/redis/issues/9493
     - fix: https://github.com/redis/redis/pull/9733

    So, don't use this with EVAL or EVALSHA.
    """

    __slots__ = ("commands", "node")

    def __init__(self) -> None:
        self.commands: Dict[str, Union[int, Dict[str, Any]]] = {}

    async def initialize(self, node: Optional["ClusterNode"] = None) -> None:
        if node:
            self.node = node

        commands = await self.node.execute_command("COMMAND")
        self.commands = {cmd.lower(): command for cmd, command in commands.items()}

    # As soon as this PR is merged into Redis, we should reimplement
    # our logic to use COMMAND INFO changes to determine the key positions
    # https://github.com/redis/redis/pull/8324
    async def get_keys(self, *args: Any) -> Optional[Tuple[str, ...]]:
        """
        Get the keys from the passed command.

        Keys of 'movablekeys' commands are extracted locally from the command's
        key specifications when they are complete, falling back to
        'COMMAND GETKEYS' otherwise.

        NOTE: Due to a bug in redis<7.0, this function does not work properly
        for EVAL or EVALSHA when the `numkeys` arg is 0.
         - issue: https://github.com/redis/redis/issues/9493
         - fix: https://github.com/redis/redis/pull/9733

        So, don't use this function with EVAL or EVALSHA.
        """
        if len(args) < 2:
            # The command has no keys in it
            return None

        cmd_name = args[0].lower()
        if cmd_name not in self.commands:
            # try to split the command name and to take only the main command,
            # e.g. 'memory' for 'memory usage'
            cmd_name_split = cmd_name.split()
            cmd_name = cmd_name_split[0]
            if cmd_name in self.commands:
                # save the split command to args
                args = cmd_name_split + list(args[1:])
            else:
                # We'll try to reinitialize the commands cache, if the engine
                # version has changed, the commands may not be current
                await self.initialize()
                if cmd_name not in self.commands:
                    raise RedisError(
                        f"{cmd_name.upper()} command doesn't exist in Redis commands"
                    )

        command = self.commands.get(cmd_name)
        # A container subcommand carries its own flags and key specifications, so
        # resolve it before deciding how the keys are found.
        effective_command = command
        if len(args) > 1 and command.get("subcommands"):
            subcmd_name = f"{cmd_name}|{args[1].lower()}"
            for subcmd in command["subcommands"]:
                if str_if_bytes(subcmd[0]) == subcmd_name:
                    effective_command = self.parse_subcommand(subcmd)
                    break
        if "movablekeys" in effective_command["flags"]:
            extracted = _extract_keys_from_specs(effective_command, args)
            if extracted is not None:
                # Arguments flagged not_key (e.g. a shard pubsub channel) are not
                # key names and must not decide the slot.
                keys = [key for key, flags in extracted if "not_key" not in flags]
            else:
                keys = await self._get_moveable_keys(*args)
        elif "pubsub" in command["flags"] or command["name"] == "pubsub":
            keys = self._get_pubsub_keys(*args)
        else:
            if (
                command["step_count"] == 0
                and command["first_key_pos"] == 0
                and command["last_key_pos"] == 0
            ):
                is_subcmd = False
                if "subcommands" in command:
                    subcmd_name = f"{cmd_name}|{args[1].lower()}"
                    for subcmd in command["subcommands"]:
                        if str_if_bytes(subcmd[0]) == subcmd_name:
                            command = self.parse_subcommand(subcmd)

                            if command["first_key_pos"] > 0:
                                is_subcmd = True

                # The command doesn't have keys in it
                if not is_subcmd:
                    return None
            last_key_pos = command["last_key_pos"]
            if last_key_pos < 0:
                last_key_pos = len(args) - abs(last_key_pos)
            keys_pos = list(
                range(command["first_key_pos"], last_key_pos + 1, command["step_count"])
            )
            keys = [args[pos] for pos in keys_pos]

        return keys

    async def _get_moveable_keys(self, *args: Any) -> Optional[Tuple[str, ...]]:
        try:
            keys = await self.node.execute_command("COMMAND GETKEYS", *args)
        except ResponseError as e:
            message = e.__str__()
            if (
                "Invalid arguments" in message
                or "The command has no key arguments" in message
            ):
                return None
            else:
                raise e
        return keys

    @deprecated_function(
        version="8.2.0",
        reason="Use get_commands_metadata_cache() instead.",
    )
    async def get_command_policies(self) -> PolicyRecords:
        """
        Retrieve and process the command policies for all commands and subcommands.

        DEPRECATED: use :meth:`get_commands_metadata_cache` instead. Nothing in this library calls this
        method any more, and it will be removed in a future release. A metadata resolver keeps a
        compatibility shim for an object that serves only this method, because it was the whole
        contract in 7.1.0; that shim is not an invitation to write one. To decide routing
        yourself, implement the public ``PolicyResolver`` ABC instead.

        This method traverses through commands and subcommands, extracting policy details
        from associated data structures and constructing a dictionary of commands with their
        associated policies. It supports nested data structures and handles both main commands
        and their subcommands.

        This is the routing view of :meth:`get_commands_metadata_cache`: the same traversal of the
        same reply, keyed the same way, projected down to the two policies the cluster client
        routes by. A caller that also needs the client-side-caching metadata asks for the
        metadata records instead.

        Returns:
            PolicyRecords: A collection of commands and subcommands associated with their
            respective policies.

        Raises:
            IncorrectPolicyType: If an invalid policy type is encountered during policy extraction.
        """
        return _build_policy_records(self.commands)

    async def get_commands_metadata_cache(self) -> CommandMetadataRecordsCache:
        """
        Retrieve and process the metadata records cache for all commands and subcommands.

        This method normalizes the command flags, command tips and key metadata of the
        ``COMMAND`` output into metadata records, keyed the same way as the policy records
        that the deprecated ``get_command_policies`` returns. The routing policies each
        record carries are the ones that method resolves, so both views of the same reply
        stay in step.

        Returns:
            CommandMetadataRecordsCache: A collection of commands and subcommands
            associated with their respective metadata.

        Raises:
            IncorrectPolicyType: If an invalid policy type is encountered during policy extraction.
        """
        return _build_commands_metadata_cache(self.commands)


# =============================================================================
# Private helpers
# =============================================================================
def _parse_subcommand(command: Any) -> Dict[str, Any]:
    """
    Parse a single entry of a command's ``subcommands`` field into a details dict
    with the same shape as a top-level ``COMMAND`` entry.
    """
    cmd_dict = {}
    cmd_name = str_if_bytes(command[0])
    cmd_dict["name"] = cmd_name
    cmd_dict["arity"] = int(command[1])
    cmd_dict["flags"] = [str_if_bytes(flag) for flag in command[2]]
    cmd_dict["first_key_pos"] = command[3]
    cmd_dict["last_key_pos"] = command[4]
    cmd_dict["step_count"] = command[5]
    if len(command) > 7:
        cmd_dict["tips"] = command[7]
        cmd_dict["key_specifications"] = command[8]
        cmd_dict["subcommands"] = command[9]
    return cmd_dict


def _is_keyless_command(
    commands: Dict[str, Any],
    command_name: str,
    subcommand_name: Optional[str] = None,
) -> bool:
    """
    Determines whether a given command or subcommand is considered "keyless".

    A keyless command does not operate on specific keys, which is determined based
    on the first key position in the command or subcommand details. If the command
    or subcommand's first key position is zero or negative, it is treated as keyless.

    Parameters:
        commands: Dict[str, Any]
            The parsed ``COMMAND`` output to look the command up in.
        command_name: str
            The name of the command to check.
        subcommand_name: Optional[str], default=None
            The name of the subcommand to check, if applicable. If not provided,
            the check is performed only on the command.

    Returns:
        bool
            True if the specified command or subcommand is considered keyless,
            False otherwise.

    Raises:
        ValueError
            If the specified subcommand is not found within the command or the
            specified command does not exist in the available commands.
    """
    if subcommand_name:
        for subcommand in commands.get(command_name)["subcommands"]:
            if str_if_bytes(subcommand[0]) == subcommand_name:
                parsed_subcmd = _parse_subcommand(subcommand)
                return parsed_subcmd["first_key_pos"] <= 0
        raise ValueError(
            f"Subcommand {subcommand_name} not found in command {command_name}"
        )
    else:
        command_details = commands.get(command_name, None)
        if command_details is not None:
            return command_details["first_key_pos"] <= 0

        raise ValueError(f"Command {command_name} not found in commands")


# Slots of the two-element buffer ``_apply_policy_tips`` writes into. A list rather than a
# record, so one buffer can be reused for every command of a ``COMMAND`` reply instead of a
# policy object being allocated per command and thrown away.
_REQUEST_POLICY = 0
_RESPONSE_POLICY = 1


def _walk_tips(data: Any) -> Iterator[str]:
    """
    Recursively yield every tip string of a ``COMMAND`` reply fragment, decoding bytes.

    The one traversal both tip readers share, so the routing view and the metadata view
    cannot disagree about which structures a tip may be nested inside. That matters because a
    metadata record is the superset of a policy record: it carries the two routing policies
    plus the cacheability markers, so it has to read at least everything the policy view
    reads off the same field.

    A ``COMMAND`` reply reports a command's ``tips`` as a flat array, but the fragments
    :func:`_apply_policy_tips` is handed for a container subcommand are whole reply entries,
    whose nested arrays and maps have to be walked. Anything that is neither a string, an
    array nor a map carries no tip and is skipped.

    Args:
        data: The fragment to walk (can be list, dict, str, bytes, etc.)

    Yields:
        Each tip string found, in reply order, so that a caller keeping only one value per
        slot ends up with the last one the reply carried.
    """
    if isinstance(data, (str, bytes)):
        # Decode bytes to string if needed
        yield str_if_bytes(data)

    elif isinstance(data, list):
        # For lists, recursively process each element
        for item in data:
            yield from _walk_tips(item)

    elif isinstance(data, dict):
        # For dictionaries, recursively process each value
        for value in data.values():
            yield from _walk_tips(value)


def _apply_policy_tips(policy_pair: List[Any], data: Any) -> None:
    """
    Extract policies from nested data structures.

    Args:
        policy_pair: The ``[request_policy, response_policy]`` buffer to update in place.
            The last policy tip found for a slot wins.
        data: The data structure to search, walked by :func:`_walk_tips`.

    Raises:
        IncorrectPolicyType: If an invalid policy type is encountered.
    """
    for policy in _walk_tips(data):
        # Check if this is a policy string
        if policy.startswith("request_policy"):
            policy_type = policy.split(":")[1]

            try:
                policy_pair[_REQUEST_POLICY] = RequestPolicy(policy_type)
            except ValueError:
                raise IncorrectPolicyType(
                    f"Incorrect request policy type: {policy_type}"
                )

        elif policy.startswith("response_policy"):
            policy_type = policy.split(":")[1]

            try:
                policy_pair[_RESPONSE_POLICY] = ResponsePolicy(policy_type)
            except ValueError:
                raise IncorrectPolicyType(
                    f"Incorrect response policy type: {policy_type}"
                )


def _build_command_records(
    commands: Dict[str, Any],
    to_record: Callable[[Dict[str, Any], RequestPolicy, ResponsePolicy], _RecordT],
) -> Dict[str, Dict[str, _RecordT]]:
    """
    Traverse a parsed ``COMMAND`` reply, resolving the policies of every command, and build
    one record per command with ``to_record``.

    ``to_record`` is called once per command and once per container subcommand, with the
    command's details and its resolved policies. Policies start at the keyed or keyless
    defaults, chosen by whether the command takes keys, and are then overwritten by the
    ``request_policy`` and ``response_policy`` command tips.

    This is the single traversal behind both the policy records and the metadata records, so
    the two are keyed identically by construction and the tips of a reply are walked once no
    matter which view a caller asks for. The policies are handed over as plain enum members,
    so the caller that wants a ``CommandPolicies`` record builds one and the caller that
    wants a ``CommandMetadata`` record is not charged for one it would discard.

    Args:
        commands: The parsed ``COMMAND`` output to traverse.
        to_record: Builds the record for one command from its details and policies.

    Returns:
        The records of every command and subcommand, keyed by module name, then by command
        name.

    Raises:
        IncorrectPolicyType: If an invalid policy type is encountered during policy extraction.
    """
    command_records: Dict[str, Dict[str, _RecordT]] = {}
    # Reused by every command below: the policies are read out of it before the next command
    # resets it, so no per-command buffer is allocated.
    policy_pair: List[Any] = [
        RequestPolicy.DEFAULT_KEYLESS,
        ResponsePolicy.DEFAULT_KEYLESS,
    ]

    for command, details in commands.items():
        # Check whether the command has keys
        is_keyless = _is_keyless_command(commands, command)

        if is_keyless:
            policy_pair[_REQUEST_POLICY] = RequestPolicy.DEFAULT_KEYLESS
            policy_pair[_RESPONSE_POLICY] = ResponsePolicy.DEFAULT_KEYLESS
        else:
            policy_pair[_REQUEST_POLICY] = RequestPolicy.DEFAULT_KEYED
            policy_pair[_RESPONSE_POLICY] = ResponsePolicy.DEFAULT_KEYED

        module_name, command_name = _split_module_and_command(command)
        module_records = command_records.setdefault(module_name, {})

        tips = details.get("tips")
        subcommands = details.get("subcommands")

        # Process tips for the main command
        if tips:
            _apply_policy_tips(policy_pair, tips)

        module_records[command_name] = to_record(
            details, policy_pair[_REQUEST_POLICY], policy_pair[_RESPONSE_POLICY]
        )

        # Process subcommands
        if subcommands:
            for subcommand_details in subcommands:
                # Get the subcommand name (first element)
                subcmd_name = subcommand_details[0]
                if isinstance(subcmd_name, bytes):
                    subcmd_name = subcmd_name.decode()

                # Check whether the subcommand has keys
                is_keyless = _is_keyless_command(commands, command, subcmd_name)

                if is_keyless:
                    policy_pair[_REQUEST_POLICY] = RequestPolicy.DEFAULT_KEYLESS
                    policy_pair[_RESPONSE_POLICY] = ResponsePolicy.DEFAULT_KEYLESS
                else:
                    policy_pair[_REQUEST_POLICY] = RequestPolicy.DEFAULT_KEYED
                    policy_pair[_RESPONSE_POLICY] = ResponsePolicy.DEFAULT_KEYED

                # Container subcommands are keyed by their space-joined name, e.g.
                # ``memory usage``, which is the form ``execute_command`` receives.
                subcmd_name = subcmd_name.replace("|", " ")

                # Recursively extract policies from the rest of the subcommand details
                for subcommand_detail in subcommand_details[1:]:
                    _apply_policy_tips(policy_pair, subcommand_detail)

                module_records[subcmd_name] = to_record(
                    _parse_subcommand(subcommand_details),
                    policy_pair[_REQUEST_POLICY],
                    policy_pair[_RESPONSE_POLICY],
                )

    return command_records


def _to_command_policies(
    details: Dict[str, Any],
    request_policy: RequestPolicy,
    response_policy: ResponsePolicy,
) -> CommandPolicies:
    """Build the policy record for a single entry of a ``COMMAND`` reply."""
    return CommandPolicies(
        request_policy=request_policy, response_policy=response_policy
    )


def _build_policy_records(commands: Dict[str, Any]) -> PolicyRecords:
    """
    Retrieve and process the command policies for all commands and subcommands.

    This function traverses through commands and subcommands, extracting policy details
    from associated data structures and constructing a dictionary of commands with their
    associated policies. It supports nested data structures and handles both main commands
    and their subcommands.

    Args:
        commands: The parsed ``COMMAND`` output to build the policy records from.

    Returns:
        PolicyRecords: A collection of commands and subcommands associated with their
        respective policies.

    Raises:
        IncorrectPolicyType: If an invalid policy type is encountered during policy extraction.
    """
    return _build_command_records(commands, _to_command_policies)


def _split_module_and_command(command: str) -> Tuple[str, str]:
    """
    Split a name from a ``COMMAND`` reply into the ``(module, command)`` pair the record
    tables are keyed by. Non-module commands live under ``"core"``.
    """
    split_name = command.split(".")

    if len(split_name) > 1:
        return split_name[0], split_name[1]

    return "core", split_name[0]


def _key_spec_flags(key_spec: Any) -> Set[str]:
    """
    The flags of a single key specification.

    RESP3 reports a key spec as a map and RESP2 as a flat ``[name, value, ...]`` sequence,
    so both shapes are read here.
    """
    if isinstance(key_spec, dict):
        flags = key_spec.get(b"flags") or key_spec.get("flags") or ()
    else:
        flags = ()
        for index in range(0, len(key_spec) - 1, 2):
            if str_if_bytes(key_spec[index]) == "flags":
                flags = key_spec[index + 1]
                break

    return {str_if_bytes(flag) for flag in flags}


def _has_key_argument(details: Dict[str, Any]) -> bool:
    """
    Whether the command accepts at least one Redis key name argument.

    Key specs decide it whenever the server reports them, including for ``movablekeys``
    commands, whose keys are not discoverable from the legacy positions at all. A spec
    flagged ``not_key`` describes an argument that is not a key name, such as a shard
    pubsub channel, so a command whose every spec is ``not_key`` takes no keys.

    Only a server that reports no key specs falls back to the legacy positions.
    ``last_key_pos`` is never consulted: it is ``-1`` for variadic commands whose key
    arguments run to the end of the argument list, and must not disqualify one.
    """
    key_specs = details.get("key_specifications")

    if key_specs:
        return any("not_key" not in _key_spec_flags(spec) for spec in key_specs)

    return details.get("first_key_pos", 0) > 0 and details.get("step_count", 0) > 0


def _to_command_metadata(
    details: Dict[str, Any],
    request_policy: RequestPolicy,
    response_policy: ResponsePolicy,
) -> CommandMetadata:
    """
    Build the metadata record for a single entry of a ``COMMAND`` reply.

    ``details`` is either a top-level entry or a ``_parse_subcommand`` result. The routing
    policies are passed in rather than derived again, so a record always carries the same
    policies ``_build_policy_records`` resolves for the command.
    """
    flags = {str_if_bytes(flag) for flag in details.get("flags") or ()}
    # Walked rather than read as a flat sequence, through the same traversal the routing view
    # resolves its tips with: a metadata record is the superset of a policy record, so it must
    # not read less of the field than the policy view does.
    tips = set(_walk_tips(details.get("tips")))

    return CommandMetadata(
        request_policy=request_policy,
        response_policy=response_policy,
        is_readonly="readonly" in flags,
        is_blocking="blocking" in flags,
        has_key_argument=_has_key_argument(details),
        # Matched exactly: ``nondeterministic_output_order`` is a different tip, denoting
        # only that element order varies.
        has_nondeterministic_output="nondeterministic_output" in tips,
        is_script_runner="script_runner" in flags,
        is_dont_cache="dont_cache" in tips,
        # An empty tips list is a real answer; a missing tips key means the server is too
        # old to report tips at all, which makes the negative markers undetectable.
        has_complete_metadata="flags" in details and "tips" in details,
    )


def _build_commands_metadata_cache(
    commands: Dict[str, Any],
) -> CommandMetadataRecordsCache:
    """
    Retrieve and process the metadata records cache for all commands and subcommands.

    This function traverses through commands and subcommands, normalizing the command
    flags, command tips and key metadata of each into a metadata record. It shares its
    traversal with ``_build_policy_records``, so records are keyed the same way and carry
    the same policies, and a metadata resolver and a policy resolver agree on the same reply.

    Args:
        commands: The parsed ``COMMAND`` output to build the metadata records from.

    Returns:
        CommandMetadataRecordsCache: A collection of commands and subcommands associated
        with their respective metadata.

    Raises:
        IncorrectPolicyType: If an invalid policy type is encountered during policy extraction.
    """
    return _build_command_records(commands, _to_command_metadata)


# =============================================================================
# Local key extraction from key specifications
# =============================================================================
# Redis 7 key specifications describe where a command's key name arguments sit, which
# makes the keys of most ``movablekeys`` commands computable on the client, without the
# ``COMMAND GETKEYS`` round trip. The extraction below mirrors the server's
# ``getKeysUsingKeySpec``: ``begin_search`` finds where the keys start (a fixed ``index``
# or the argument after a ``keyword``), and ``find_keys`` finds where they end (a
# ``range`` or a ``keynum`` counter). Anything the specs do not fully describe - an
# ``unknown`` begin_search/find_keys, an ``incomplete`` spec flag, a shape this client
# does not implement - makes the whole extraction return None, and the caller falls back
# to ``COMMAND GETKEYS``: a command routed by keys that were guessed wrong would reach
# an arbitrary node.

# Maps only ASCII upper case to lower case, which is the case folding the server's
# ``strcasecmp`` applies to key-spec keywords. ``str.lower`` would also fold non-ASCII
# letters, matching arguments the server would not.
_ASCII_LOWER_TABLE = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"
)


def _spec_fields(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize a key-specification fragment to a string-keyed dict.

    RESP3 reports key specs as maps and RESP2 as flat ``[name, value, ...]`` sequences,
    so both shapes are read here. Returns None for anything else - notably the
    ``"unknown"`` string a server reports for a spec it cannot describe.
    """
    if isinstance(raw, dict):
        return {str_if_bytes(key): value for key, value in raw.items()}
    if isinstance(raw, (list, tuple)) and len(raw) % 2 == 0:
        return {
            str_if_bytes(raw[index]): raw[index + 1] for index in range(0, len(raw), 2)
        }
    return None


def _normalize_key_spec(
    raw_spec: Any,
) -> Optional[Tuple[FrozenSet[str], Tuple[str, Any], Tuple[str, Any]]]:
    """
    Normalize one raw key specification into ``(flags, begin_search, find_keys)``.

    ``begin_search`` is ``("index", position)`` or ``("keyword", (word, startfrom))``;
    ``find_keys`` is ``("range", (lastkey, keystep, limit))`` or
    ``("keynum", (keynum, firstkey, keystep))``.

    Returns None when the spec is flagged ``incomplete``, uses a mechanism this client
    does not implement, or is malformed - all of which the caller treats as "resolve
    the keys on the server instead".
    """
    fields = _spec_fields(raw_spec)
    if fields is None:
        return None

    flags = _key_spec_flags(raw_spec)
    # An incomplete spec does not describe all of the command's keys (e.g. MIGRATE,
    # whose leading key argument is a key only when KEYS is absent); extracting the
    # keys it does describe would route the command by a partial key list.
    if "incomplete" in flags:
        return None

    begin_search = _spec_fields(fields.get("begin_search"))
    if begin_search is None or len(begin_search) != 1:
        return None
    if "index" in begin_search:
        index = _spec_fields(begin_search["index"])
        if index is None or not isinstance(index.get("index"), int):
            return None
        begin: Tuple[str, Any] = ("index", index["index"])
    elif "keyword" in begin_search:
        keyword = _spec_fields(begin_search["keyword"])
        if keyword is None:
            return None
        word = keyword.get("keyword")
        startfrom = keyword.get("startfrom")
        if not isinstance(word, (str, bytes)) or not isinstance(startfrom, int):
            return None
        begin = ("keyword", (str_if_bytes(word), startfrom))
    else:
        return None

    find_keys = _spec_fields(fields.get("find_keys"))
    if find_keys is None or len(find_keys) != 1:
        return None
    if "range" in find_keys:
        range_fields = _spec_fields(find_keys["range"])
        if range_fields is None:
            return None
        lastkey = range_fields.get("lastkey")
        keystep = range_fields.get("keystep")
        limit = range_fields.get("limit")
        if not all(isinstance(value, int) for value in (lastkey, keystep, limit)):
            return None
        if keystep <= 0 or limit < 0:
            return None
        find: Tuple[str, Any] = ("range", (lastkey, keystep, limit))
    elif "keynum" in find_keys:
        keynum = _spec_fields(find_keys["keynum"])
        if keynum is None:
            return None
        keynum_offset = keynum.get("keynum")
        firstkey = keynum.get("firstkey")
        keystep = keynum.get("keystep")
        if not all(
            isinstance(value, int) for value in (keynum_offset, firstkey, keystep)
        ):
            return None
        if keynum_offset < 0 or firstkey < 0 or keystep <= 0:
            return None
        find = ("keynum", (keynum_offset, firstkey, keystep))
    else:
        return None

    return frozenset(flags), begin, find


def _find_keyword(args: Sequence[Any], word: str, startfrom: int) -> Optional[int]:
    """
    Find the position of a key-spec keyword among the command arguments.

    A non-negative ``startfrom`` scans forward from that position; a negative one
    scans backward from that many arguments before the end, so the last occurrence
    wins - both exactly as the server searches. Returns None when the keyword is
    absent.
    """
    word = word.translate(_ASCII_LOWER_TABLE)
    word_bytes = word.encode()
    if startfrom >= 0:
        positions = range(startfrom, len(args))
    else:
        positions = range(len(args) + startfrom, -1, -1)
    for position in positions:
        arg = args[position]
        if isinstance(arg, bytes):
            # bytes.lower is ASCII-only, matching the server's strcasecmp.
            if arg.lower() == word_bytes:
                return position
        elif isinstance(arg, str):
            if arg.translate(_ASCII_LOWER_TABLE) == word:
                return position
    return None


def _parse_numkeys(value: Any) -> Optional[int]:
    """
    Parse a ``keynum`` counter argument with the strictness of the server's
    ``string2ll``: an optional leading minus followed by ASCII digits, nothing else.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        digits = value[1:] if value.startswith("-") else value
        if digits and digits.isascii() and digits.isdigit():
            return int(value)
    return None


def _extract_keys_from_specs(
    details: Dict[str, Any], args: Sequence[Any]
) -> Optional[List[Tuple[Any, FrozenSet[str]]]]:
    """
    Extract the key name arguments of a command invocation from its key
    specifications, without asking the server.

    Args:
        details: The parsed ``COMMAND`` entry of the command or container
            subcommand, carrying its ``key_specifications``.
        args: The full command invocation, with ``args[0]`` the command name, in
            the shape the server would receive it.

    Returns:
        A list of ``(key, flags)`` pairs - one per key name argument, in argument
        order, each carrying the access flags of the key spec that produced it -
        or None when the specs do not describe the command's keys completely and
        the keys must be resolved on the server instead. Arguments flagged
        ``not_key`` are included with their flag; the caller decides whether they
        participate in slot determination.

    Raises:
        ResponseError: When the invocation is malformed for a known spec - a
            ``keynum`` counter that is missing, not a non-negative integer, or
            counts more keys than the argument list holds, or a fixed key position
            beyond the argument list. Reported locally with the ``Invalid
            arguments`` error ``COMMAND GETKEYS`` would produce, so the failure
            surfaces before the command is routed and sent.
    """
    key_specs = details.get("key_specifications")
    if not key_specs:
        return None

    keys: List[Tuple[Any, FrozenSet[str]]] = []
    for raw_spec in key_specs:
        spec = _normalize_key_spec(raw_spec)
        if spec is None:
            return None
        flags, (begin_type, begin), (find_type, find) = spec

        if begin_type == "index":
            start = begin
            if start < 0:
                start += len(args)
            if start < 0 or start >= len(args):
                raise ResponseError("Invalid arguments")
        else:
            word, startfrom = begin
            position = _find_keyword(args, word, startfrom)
            if position is None:
                # The keyword is absent: the optional argument group it opens was
                # not given, so this spec contributes no keys.
                continue
            start = position + 1

        if find_type == "range":
            lastkey, keystep, limit = find
            if lastkey >= 0:
                last = start + lastkey
                if last >= len(args):
                    raise ResponseError("Invalid arguments")
            else:
                last = len(args) + lastkey
                if limit:
                    # The trailing arguments come in groups of ``limit`` and only
                    # the first of each group is a key (e.g. XREAD, where stream
                    # names and ids alternate).
                    last = start + (last - start + 1) // limit - 1
            step = keystep
        else:
            keynum_offset, firstkey, keystep = find
            numkeys_position = start + keynum_offset
            if numkeys_position >= len(args):
                raise ResponseError("Invalid arguments")
            numkeys = _parse_numkeys(args[numkeys_position])
            if numkeys is None or numkeys < 0:
                raise ResponseError("Invalid arguments")
            start += firstkey
            last = start + (numkeys - 1) * keystep
            if numkeys and last >= len(args):
                raise ResponseError("Invalid arguments")
            step = keystep

        for position in range(start, last + 1, step):
            keys.append((args[position], flags))

    return keys
