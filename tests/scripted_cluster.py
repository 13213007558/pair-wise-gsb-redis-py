"""
A scripted three-node fake Redis cluster for tests.

Each node is a scripted peer the client talks to over a dedicated socketpair,
so no real ports are bound while the wire protocol, connection lifecycle and
retry paths stay entirely real. Nodes speak just enough RESP for the redis-py
cluster client (HELLO / CLIENT / CLUSTER SLOTS / ASKING / PING), while MGET
replies are scripted per test. Every node records the commands it receives so
tests can assert on request content, ordering and network call counts.

The cluster client reaches the nodes through the custom connection classes
exposed as ``ScriptedCluster.connection_class`` (sync) and
``ScriptedCluster.async_connection_class`` (async): instead of dialing TCP,
they open a socketpair to the scripted node owning the advertised
host/port. CLUSTER SLOTS advertises fixed fake addresses, and MOVED/ASK
replies redirect between them.
"""

import asyncio
import socket
import threading

from redis.asyncio.connection import Connection as AsyncConnection
from redis.connection import Connection
from redis.crc import key_slot


def _bulk_string(value: bytes) -> bytes:
    return b"$" + str(len(value)).encode() + b"\r\n" + value + b"\r\n"


def _array(items) -> bytes:
    return b"*" + str(len(items)).encode() + b"\r\n" + b"".join(items)


def _integer(value: int) -> bytes:
    return b":" + str(value).encode() + b"\r\n"


def mget_reply(values, resp3: bool = False) -> bytes:
    """Build an MGET array reply; ``None`` entries become null replies in
    the dialect of the connection's protocol (``_`` for RESP3, ``$-1`` for
    RESP2), mirroring what a real server sends."""
    nil = b"_\r\n" if resp3 else b"$-1\r\n"
    return _array([nil if value is None else _bulk_string(value) for value in values])


def error_reply(message: bytes) -> bytes:
    return b"-" + message + b"\r\n"


def moved_reply(slot: int, host: bytes, port: int) -> bytes:
    return error_reply(b"MOVED %d %s:%d" % (slot, host, port))


def ask_reply(slot: int, host: bytes, port: int) -> bytes:
    return error_reply(b"ASK %d %s:%d" % (slot, host, port))


def command_reply() -> bytes:
    """A ``COMMAND`` reply carrying just the MGET command table entry."""
    mget_entry = _array(
        [
            _bulk_string(b"mget"),
            _integer(-2),
            _array([_bulk_string(b"readonly"), _bulk_string(b"fast")]),
            _integer(1),
            _integer(-1),
            _integer(1),
            _array([]),  # ACL categories
            _array([]),  # tips
            _array([]),  # key specifications
            _array([]),  # subcommands
        ]
    )
    return _array([mget_entry])


def drop_connection(keys) -> None:
    """An MGET script that drops the connection without replying."""
    return None


def _hello_reply(proto: bytes) -> bytes:
    return (
        b"%2\r\n"
        b"$5\r\nproto\r\n:" + proto + b"\r\n"
        b"$7\r\nversion\r\n" + _bulk_string(b"8.0.0")
    )


def _read_command(reader):
    """Read one RESP array of bulk strings; return None on EOF."""
    line = reader.readline()
    if not line:
        return None
    count = int(line[1:])
    args = []
    for _ in range(count):
        header = reader.readline()
        length = int(header[1:])
        args.append(reader.read(length))
        reader.read(2)  # trailing CRLF
    return args


class ScriptedClusterNode:
    """One fake cluster node with a scriptable MGET reply."""

    def __init__(self, cluster: "ScriptedCluster", port: int, node_id: bytes):
        self.cluster = cluster
        self.node_id = node_id
        self.host = "127.0.0.1"
        self.port = port
        # Callable taking the MGET keys (list of bytes) and returning either
        # a list of values (bytes or None) for the node to encode, a raw
        # RESP reply as bytes, or None to drop the connection without
        # replying.
        self.mget_script = lambda keys: [None] * len(keys)
        self._requests = []
        self._server_sockets = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"{self.host}:{self.port}"

    def attach(self, server_sock: socket.socket):
        """Serve one client connection on its own thread."""
        with self._lock:
            self._server_sockets.append(server_sock)
        thread = threading.Thread(target=self._serve, args=(server_sock,))
        thread.daemon = True
        thread.start()

    def _serve(self, server_sock: socket.socket):
        try:
            reader = server_sock.makefile("rb")
            proto = 2
            while True:
                args = _read_command(reader)
                if args is None:
                    return
                self.record(args)
                reply = self.dispatch(args, proto)
                if args[0].upper() == b"HELLO":
                    proto = int(args[1])
                if reply is None:
                    # Scripted connection drop: close without replying.
                    return
                server_sock.sendall(reply)
        except OSError:
            # The client went away mid-conversation.
            return
        finally:
            try:
                server_sock.close()
            except OSError:
                pass

    def close_connections(self):
        with self._lock:
            sockets, self._server_sockets = self._server_sockets, []
        for server_sock in sockets:
            try:
                server_sock.close()
            except OSError:
                pass

    def record(self, args):
        with self._lock:
            self._requests.append(args)

    @property
    def requests(self):
        with self._lock:
            return list(self._requests)

    def received(self, command: bytes):
        """All recorded commands matching ``command`` (e.g. ``b"MGET"``)."""
        return [args for args in self.requests if args[0].upper() == command]

    def dispatch(self, args, proto: int):
        command = args[0].upper()
        if command == b"HELLO":
            return _hello_reply(args[1])
        if command in (b"CLIENT", b"SELECT", b"READONLY"):
            return b"+OK\r\n"
        if command == b"PING":
            return b"+PONG\r\n"
        if command == b"ASKING":
            return b"+OK\r\n"
        if command == b"CLUSTER" and args[1].upper() == b"SLOTS":
            return self.cluster.cluster_slots_reply()
        if command == b"COMMAND" and len(args) == 1:
            return command_reply()
        if command == b"MGET":
            reply = self.mget_script(args[1:])
            if isinstance(reply, list):
                reply = mget_reply(reply, resp3=proto == 3)
            return reply
        return error_reply(b"ERR unknown command `%s`" % command)


class ScriptedCluster:
    """Three primaries covering all 16384 slots, one range per node."""

    SLOT_RANGES = ((0, 5460), (5461, 10922), (10923, 16383))
    PORTS = (30001, 30002, 30003)

    def __init__(self):
        self.nodes = [
            ScriptedClusterNode(self, port, b"node-%d" % index)
            for index, port in enumerate(self.PORTS)
        ]
        cluster = self

        class ScriptedConnection(Connection):
            """Connects to the scripted node owning this host/port."""

            def _connect(self):
                node = cluster.node_by_address(self.host, self.port)
                client_sock, server_sock = socket.socketpair()
                node.attach(server_sock)
                if self.socket_timeout is not None:
                    client_sock.settimeout(self.socket_timeout)
                return client_sock

        class AsyncScriptedConnection(AsyncConnection):
            """Async counterpart of ``ScriptedConnection``."""

            async def _connect(self):
                node = cluster.node_by_address(self.host, self.port)
                client_sock, server_sock = socket.socketpair()
                node.attach(server_sock)
                reader, writer = await asyncio.open_connection(sock=client_sock)
                self._reader = reader
                self._writer = writer

        self.connection_class = ScriptedConnection
        self.async_connection_class = AsyncScriptedConnection

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        for node in self.nodes:
            node.close_connections()

    def startup_nodes(self):
        from redis.cluster import ClusterNode

        return [ClusterNode(self.nodes[0].host, self.nodes[0].port)]

    def node_by_address(self, host: str, port: int) -> ScriptedClusterNode:
        for node in self.nodes:
            if node.host == host and node.port == int(port):
                return node
        raise AssertionError(f"no scripted node at {host}:{port}")

    def node_for_key(self, key: bytes) -> ScriptedClusterNode:
        slot = key_slot(key)
        for (start, end), node in zip(self.SLOT_RANGES, self.nodes):
            if start <= slot <= end:
                return node
        raise AssertionError(f"no node covers slot {slot}")

    def cluster_slots_reply(self) -> bytes:
        entries = []
        for (start, end), node in zip(self.SLOT_RANGES, self.nodes):
            primary = _array(
                [
                    _bulk_string(node.host.encode()),
                    _integer(node.port),
                    _bulk_string(node.node_id),
                ]
            )
            entries.append(_array([_integer(start), _integer(end), primary]))
        return _array(entries)


def keys_for_each_node(cluster: ScriptedCluster, tag: bytes = b"key"):
    """Pick one hash-tagged key per node, in node order."""
    keys = []
    candidate = 0
    for node in cluster.nodes:
        while True:
            key = b"{%s-%d}" % (tag, candidate)
            candidate += 1
            if cluster.node_for_key(key) is node:
                keys.append(key)
                break
    return keys
