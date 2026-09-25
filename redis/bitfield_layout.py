"""
Declarative bit-packing layouts on top of Redis ``BITFIELD``.

A :class:`BitFieldLayout` describes how named, fixed-width integer fields
(signed ``iN`` / unsigned ``uN`` encodings) and explicit padding gaps are
packed into fixed-stride records inside a single Redis string, so callers
can read and write named values by record number instead of hand-computing
``BITFIELD`` bit offsets.  Bit order and offset semantics are exactly those
of the Redis ``BITFIELD`` command: offsets are bit offsets from the start of
the string, most-significant bit first, and fields may span byte boundaries.

Compiling a layout validates it once (unsupported widths, overlapping
records, duplicate names) and produces an immutable, reusable
:class:`CompiledBitFieldLayout`.  Each ``read``/``write`` call builds a
fresh, self-contained plan object holding the command arguments and the
response mapping for that call, so a shared compiled layout never leaks
state between calls.

Usage::

    layout = (
        BitFieldLayout(stride=24)
        .field("score", "i5")
        .gap(3)
        .field("level", "u11")
        .compile()
    )
    plan = layout.write("stats", record=2, values={"score": -3, "level": 9})
    response = r.execute_command(*plan.command)  # await for asyncio clients
    plan.parse(response)  # {"score": <previous>, "level": <previous>}

    plan = layout.read("stats", record=2)
    response = r.execute_command(*plan.command)  # BITFIELD_RO
    plan.parse(response)  # {"score": -3, "level": 9}
"""

from dataclasses import dataclass

from redis.exceptions import DataError, InvalidResponse
from redis.typing import EncodableT, KeyT

#: Redis strings are limited to 512 MiB, i.e. 2**32 bits.
MAX_BIT_POSITION = 1 << 32

#: Highest field width Redis accepts for signed (``iN``) encodings.
MAX_SIGNED_WIDTH = 63
#: Highest field width Redis accepts for unsigned (``uN``) encodings.
MAX_UNSIGNED_WIDTH = 64

_OVERFLOW_MODES = ("WRAP", "SAT", "FAIL")


@dataclass(frozen=True)
class BitField:
    """A single named field of a compiled layout record."""

    name: str
    fmt: str  # Redis BITFIELD encoding, e.g. "i5" or "u11"
    signed: bool
    width: int
    offset: int  # bit offset of the field within its record

    @property
    def min_value(self) -> int:
        if self.signed:
            return -(1 << (self.width - 1))
        return 0

    @property
    def max_value(self) -> int:
        if self.signed:
            return (1 << (self.width - 1)) - 1
        return (1 << self.width) - 1


def _parse_fmt(fmt: str) -> tuple[bool, int]:
    """Validate a Redis BITFIELD encoding string, return (signed, width)."""
    if (
        not isinstance(fmt, str)
        or len(fmt) < 2
        or fmt[0] not in ("i", "u")
        or not fmt[1:].isdigit()
    ):
        raise DataError(
            f"invalid bitfield encoding {fmt!r}: expected 'iN' or 'uN', "
            "e.g. 'i5' or 'u11'"
        )
    signed = fmt[0] == "i"
    width = int(fmt[1:])
    limit = MAX_SIGNED_WIDTH if signed else MAX_UNSIGNED_WIDTH
    if not 1 <= width <= limit:
        raise DataError(
            f"unsupported bit width in {fmt!r}: Redis supports "
            f"i1..i{MAX_SIGNED_WIDTH} and u1..u{MAX_UNSIGNED_WIDTH}"
        )
    return signed, width


def _apply_overflow(field: BitField, value: int, overflow: str) -> int | None:
    """
    Coerce ``value`` into the range of ``field`` according to ``overflow``.

    Redis applies the ``OVERFLOW`` directive only to ``INCRBY``; ``SET``
    always truncates silently.  To give writes exact WRAP/SAT/FAIL semantics
    the coercion is done client-side and only in-range values are sent.
    Returns the value to write, or ``None`` when a FAIL check rejects it.
    """
    if overflow == "WRAP":
        masked = value & ((1 << field.width) - 1)
        if field.signed and masked > field.max_value:
            masked -= 1 << field.width
        return masked
    if overflow == "SAT":
        return min(max(value, field.min_value), field.max_value)
    # FAIL
    if field.min_value <= value <= field.max_value:
        return value
    return None


class BitFieldLayout:
    """
    Builder for a record layout.  Declare fields and gaps in bit order,
    then call :meth:`compile` to validate and freeze the layout.

    :param stride: record stride in bits; record ``n`` starts at bit
        ``n * stride``.  Defaults to the packed record size (no padding
        between records).  Must be at least the record size, otherwise
        consecutive records would overlap.
    """

    def __init__(self, stride: int | None = None):
        if stride is not None and (not isinstance(stride, int) or stride < 1):
            raise DataError("stride must be a positive integer number of bits")
        self._stride = stride
        self._parts: list[tuple] = []

    def field(self, name: str, fmt: str) -> "BitFieldLayout":
        """
        Append a named field with Redis BITFIELD encoding ``fmt``
        (``iN`` for signed, ``uN`` for unsigned, e.g. ``"i5"``/``"u11"``).
        """
        self._parts.append(("field", name, fmt))
        return self

    def gap(self, width: int) -> "BitFieldLayout":
        """Append ``width`` padding bits that are never read or written."""
        if not isinstance(width, int) or width < 1:
            raise DataError("gap width must be a positive integer number of bits")
        self._parts.append(("gap", width))
        return self

    def compile(self) -> "CompiledBitFieldLayout":
        """Validate the layout and return its immutable compiled form."""
        return CompiledBitFieldLayout(self._parts, self._stride)


class CompiledBitFieldLayout:
    """
    Immutable, reusable compiled layout.  Builds per-call ``BITFIELD`` /
    ``BITFIELD_RO`` command plans keyed by record number.
    """

    def __init__(self, parts: list[tuple], stride: int | None):
        fields = []
        names = set()
        offset = 0
        for part in parts:
            if part[0] == "gap":
                offset += part[1]
                continue
            _, name, fmt = part
            if not isinstance(name, str) or not name:
                raise DataError("field name must be a non-empty string")
            if name in names:
                raise DataError(f"duplicate field name {name!r}")
            names.add(name)
            signed, width = _parse_fmt(fmt)
            fields.append(BitField(name, fmt, signed, width, offset))
            offset += width
        if not fields:
            raise DataError("a layout must declare at least one field")
        record_bits = offset
        if stride is None:
            stride = record_bits
        elif stride < record_bits:
            raise DataError(
                f"stride {stride} is smaller than the record size "
                f"{record_bits} bits: records would overlap"
            )
        self.fields: tuple[BitField, ...] = tuple(fields)
        self.field_names: tuple[str, ...] = tuple(field.name for field in fields)
        self.record_bits = record_bits
        self.stride = stride
        self._by_name = {field.name: field for field in fields}

    def _record_base(self, record: int) -> int:
        if not isinstance(record, int) or record < 0:
            raise DataError("record must be a non-negative integer")
        base = record * self.stride
        if base + self.record_bits > MAX_BIT_POSITION:
            raise DataError(
                f"record {record} is out of bounds: bit range "
                f"{base}..{base + self.record_bits - 1} exceeds the "
                "512 MiB Redis string limit"
            )
        return base

    def _select_fields(self, fields: list[str] | None) -> tuple[BitField, ...]:
        if fields is None:
            return self.fields
        selected = []
        seen = set()
        for name in fields:
            field = self._by_name.get(name)
            if field is None:
                raise DataError(f"unknown field {name!r}")
            if name in seen:
                raise DataError(f"duplicate field {name!r} in request")
            seen.add(name)
            selected.append(field)
        if not selected:
            raise DataError("no fields selected")
        return tuple(selected)

    def read(
        self,
        key: KeyT,
        record: int = 0,
        fields: list[str] | None = None,
        use_ro: bool = True,
    ) -> "BitFieldReadPlan":
        """
        Build a plan reading ``fields`` (all of them by default) of record
        ``record`` stored at ``key``.

        Uses ``BITFIELD_RO`` by default (requires Redis >= 6.0); pass
        ``use_ro=False`` to emit a plain ``BITFIELD`` command instead, which
        is still read-only here because only ``GET`` subcommands are used.
        """
        base = self._record_base(record)
        selected = self._select_fields(fields)
        command: list[EncodableT] = ["BITFIELD_RO" if use_ro else "BITFIELD", key]
        for field in selected:
            command.extend(("GET", field.fmt, base + field.offset))
        return BitFieldReadPlan(command, tuple(field.name for field in selected))

    def write(
        self,
        key: KeyT,
        record: int = 0,
        values: dict[str, int] | None = None,
        overflow: str = "WRAP",
    ) -> "BitFieldWritePlan":
        """
        Build a plan writing the named ``values`` into record ``record``
        stored at ``key``.

        ``overflow`` is one of ``WRAP`` (truncate to the field width, the
        Redis ``SET`` behaviour), ``SAT`` (clamp to the field range) or
        ``FAIL`` (skip out-of-range fields).  It is enforced client-side
        because Redis applies ``OVERFLOW`` only to ``INCRBY``, never to
        ``SET``.  Under ``FAIL`` the fields that did not fit are left
        unwritten while the remaining fields of the record are still
        written; the parsed result maps each rejected field to ``None`` so
        it is clear exactly which fields were not written.
        """
        if not isinstance(overflow, str) or overflow.upper() not in _OVERFLOW_MODES:
            raise DataError("overflow must be one of WRAP, SAT or FAIL")
        overflow = overflow.upper()
        if not values:
            raise DataError("no values to write")
        base = self._record_base(record)
        command: list[EncodableT] = ["BITFIELD", key]
        sent = []
        skipped = []
        order = []
        for name, value in values.items():
            field = self._by_name.get(name)
            if field is None:
                raise DataError(f"unknown field {name!r}")
            if not isinstance(value, int):
                raise DataError(f"value for field {name!r} must be an integer")
            order.append(name)
            coerced = _apply_overflow(field, value, overflow)
            if coerced is None:
                skipped.append(name)
                continue
            command.extend(("SET", field.fmt, base + field.offset, coerced))
            sent.append(name)
        return BitFieldWritePlan(command, tuple(sent), tuple(skipped), tuple(order))


class BitFieldReadPlan:
    """
    Per-call read command and its response mapping.

    Execute ``plan.command`` with ``client.execute_command`` (awaiting the
    result for asyncio clients), then pass the response to :meth:`parse`.
    """

    __slots__ = ("command", "fields")

    def __init__(self, command: list[EncodableT], fields: tuple[str, ...]):
        self.command = command
        self.fields = fields

    def parse(self, response: list[int]) -> dict[str, int]:
        """Map a ``BITFIELD``/``BITFIELD_RO`` response to a fresh dict."""
        if len(response) != len(self.fields):
            raise InvalidResponse(
                f"expected {len(self.fields)} bitfield values, got {len(response)}"
            )
        return dict(zip(self.fields, response))


class BitFieldWritePlan:
    """
    Per-call write command and its response mapping.

    Execute ``plan.command`` with ``client.execute_command`` (awaiting the
    result for asyncio clients), then pass the response to :meth:`parse`.
    ``SET`` returns the previous value of each written field, so the parsed
    mapping holds previous values; fields rejected by a ``FAIL`` overflow
    check map to ``None``.
    """

    __slots__ = ("_command", "_sent", "_skipped", "_order")

    def __init__(
        self,
        command: list[EncodableT],
        sent: tuple[str, ...],
        skipped: tuple[str, ...],
        order: tuple[str, ...],
    ):
        self._command = command
        self._sent = sent
        self._skipped = skipped
        self._order = order

    @property
    def operations(self) -> int:
        """Number of ``SET`` subcommands in the command."""
        return len(self._sent)

    @property
    def skipped(self) -> tuple[str, ...]:
        """Names of fields rejected by the FAIL overflow check, if any."""
        return self._skipped

    @property
    def command(self) -> list[EncodableT]:
        if not self._sent:
            raise DataError(
                "nothing to execute: every value was rejected by the "
                "FAIL overflow check"
            )
        return self._command

    def parse(self, response: list[int]) -> dict[str, int | None]:
        """
        Map a ``BITFIELD`` response to a fresh dict, in the order the
        values were supplied.  Fields skipped by the FAIL overflow check
        map to ``None``; every other field keeps its own response slot, so
        one unwritten field never shifts the values behind it.
        """
        if len(response) != len(self._sent):
            raise InvalidResponse(
                f"expected {len(self._sent)} bitfield values, got {len(response)}"
            )
        written = dict(zip(self._sent, response))
        return {name: written.get(name) for name in self._order}
