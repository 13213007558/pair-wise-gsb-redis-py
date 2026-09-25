import pytest

from redis.bitfield_layout import (
    MAX_BIT_POSITION,
    BitFieldLayout,
    CompiledBitFieldLayout,
)
from redis.exceptions import DataError, InvalidResponse


def make_layout(stride=24):
    # i5 field, 3 explicit padding bits, u11 field: 19-bit records
    return (
        BitFieldLayout(stride=stride)
        .field("score", "i5")
        .gap(3)
        .field("level", "u11")
        .compile()
    )


def read_bits(store, offset, width):
    value = 0
    for k in range(width):
        pos = offset + k
        byte, bit = pos >> 3, 7 - (pos & 7)
        value = (value << 1) | ((store[byte] >> bit) & 1 if byte < len(store) else 0)
    return value


def write_bits(store, offset, width, value):
    needed = (offset + width + 7) >> 3
    if len(store) < needed:
        store.extend(b"\x00" * (needed - len(store)))
    for k in range(width):
        pos = offset + k
        byte, bit = pos >> 3, 7 - (pos & 7)
        if (value >> (width - 1 - k)) & 1:
            store[byte] |= 1 << bit
        else:
            store[byte] &= ~(1 << bit) & 0xFF


def decode(fmt, raw):
    width = int(fmt[1:])
    if fmt[0] == "i" and raw >= 1 << (width - 1):
        raw -= 1 << width
    return raw


def emulate_bitfield(store, command):
    # Apply a BITFIELD/BITFIELD_RO argv to a bytearray using Redis bit
    # semantics: bit offsets from the start of the string, MSB first.
    assert command[0] in ("BITFIELD", "BITFIELD_RO")
    response = []
    i = 2
    while i < len(command):
        op, fmt, offset = command[i], command[i + 1], command[i + 2]
        width = int(fmt[1:])
        if op == "GET":
            response.append(decode(fmt, read_bits(store, offset, width)))
            i += 3
        else:
            assert op == "SET"
            old = decode(fmt, read_bits(store, offset, width))
            write_bits(store, offset, width, command[i + 3] & ((1 << width) - 1))
            response.append(old)
            i += 4
    return response


class TestLayoutCompilation:
    def test_offsets_and_stride(self):
        layout = make_layout()
        assert isinstance(layout, CompiledBitFieldLayout)
        assert layout.field_names == ("score", "level")
        assert layout.record_bits == 19
        assert layout.stride == 24
        score, level = layout.fields
        assert (score.offset, score.width, score.signed) == (0, 5, True)
        assert (level.offset, level.width, level.signed) == (8, 11, False)
        assert (score.min_value, score.max_value) == (-16, 15)
        assert (level.min_value, level.max_value) == (0, 2047)

    def test_default_stride_packs_records(self):
        layout = BitFieldLayout().field("a", "u4").field("b", "u4").compile()
        assert layout.stride == 8 == layout.record_bits

    def test_stride_smaller_than_record_overlaps(self):
        with pytest.raises(DataError, match="overlap"):
            BitFieldLayout(stride=16).field("a", "u8").field("b", "u9").compile()

    @pytest.mark.parametrize("fmt", ["i0", "i64", "u0", "u65", "x8", "8", "i", ""])
    def test_unsupported_widths_and_formats(self, fmt):
        with pytest.raises(DataError):
            BitFieldLayout().field("a", fmt).compile()

    @pytest.mark.parametrize("fmt", ["i1", "i63", "u1", "u64"])
    def test_supported_widths(self, fmt):
        layout = BitFieldLayout().field("a", fmt).compile()
        assert layout.fields[0].width == int(fmt[1:])

    def test_duplicate_field_name(self):
        with pytest.raises(DataError, match="duplicate"):
            BitFieldLayout().field("a", "u4").field("a", "u4").compile()

    def test_empty_layout(self):
        with pytest.raises(DataError):
            BitFieldLayout().compile()

    @pytest.mark.parametrize("width", [0, -1, 1.5])
    def test_invalid_gap(self, width):
        with pytest.raises(DataError):
            BitFieldLayout().gap(width)

    @pytest.mark.parametrize("stride", [0, -8, 1.5])
    def test_invalid_stride(self, stride):
        with pytest.raises(DataError):
            BitFieldLayout(stride=stride)


class TestRecordBounds:
    def test_negative_record(self):
        with pytest.raises(DataError):
            make_layout().read("k", record=-1)

    def test_record_out_of_bounds(self):
        layout = make_layout()
        record = (MAX_BIT_POSITION - layout.record_bits) // layout.stride
        layout.read("k", record=record)  # last record that still fits
        with pytest.raises(DataError, match="out of bounds"):
            layout.read("k", record=record + 1)
        with pytest.raises(DataError, match="out of bounds"):
            layout.write("k", record=record + 1, values={"score": 1})


class TestCommandBuilding:
    def test_read_uses_bitfield_ro_by_default(self):
        plan = make_layout().read("k", record=1)
        assert plan.command == [
            "BITFIELD_RO",
            "k",
            "GET",
            "i5",
            24,
            "GET",
            "u11",
            32,
        ]

    def test_read_without_ro(self):
        plan = make_layout().read("k", record=0, use_ro=False)
        assert plan.command == ["BITFIELD", "k", "GET", "i5", 0, "GET", "u11", 8]

    def test_read_field_subset(self):
        plan = make_layout().read("k", record=2, fields=["level"])
        assert plan.command == ["BITFIELD_RO", "k", "GET", "u11", 56]
        assert plan.fields == ("level",)

    def test_read_unknown_or_duplicate_field(self):
        layout = make_layout()
        with pytest.raises(DataError, match="unknown field"):
            layout.read("k", fields=["nope"])
        with pytest.raises(DataError, match="duplicate"):
            layout.read("k", fields=["score", "score"])
        with pytest.raises(DataError):
            layout.read("k", fields=[])

    def test_write_command(self):
        plan = make_layout().write("k", record=1, values={"score": -3, "level": 9})
        assert plan.command == [
            "BITFIELD",
            "k",
            "SET",
            "i5",
            24,
            -3,
            "SET",
            "u11",
            32,
            9,
        ]
        assert plan.operations == 2
        assert plan.skipped == ()

    def test_write_rejects_bad_input(self):
        layout = make_layout()
        with pytest.raises(DataError):
            layout.write("k", values={})
        with pytest.raises(DataError, match="unknown field"):
            layout.write("k", values={"nope": 1})
        with pytest.raises(DataError, match="integer"):
            layout.write("k", values={"score": "1"})
        with pytest.raises(DataError, match="overflow"):
            layout.write("k", values={"score": 1}, overflow="BOGUS")


class TestOverflowModes:
    def test_wrap_truncates(self):
        plan = make_layout().write(
            "k", values={"score": 16, "level": 2048}, overflow="WRAP"
        )
        # i5: 16 wraps to -16, u11: 2048 wraps to 0
        assert plan.command == [
            "BITFIELD",
            "k",
            "SET",
            "i5",
            0,
            -16,
            "SET",
            "u11",
            8,
            0,
        ]

    def test_wrap_negative(self):
        plan = make_layout().write("k", values={"level": -1}, overflow="WRAP")
        assert plan.command == ["BITFIELD", "k", "SET", "u11", 8, 2047]

    def test_sat_clamps(self):
        plan = make_layout().write(
            "k", values={"score": -17, "level": 9999}, overflow="SAT"
        )
        assert plan.command == [
            "BITFIELD",
            "k",
            "SET",
            "i5",
            0,
            -16,
            "SET",
            "u11",
            8,
            2047,
        ]

    def test_fail_skips_out_of_range_fields(self):
        plan = make_layout().write(
            "k", values={"score": 5, "level": 2048}, overflow="fail"
        )
        assert plan.command == ["BITFIELD", "k", "SET", "i5", 0, 5]
        assert plan.skipped == ("level",)
        assert plan.parse([0]) == {"score": 0, "level": None}

    def test_fail_keeps_following_values_aligned(self):
        layout = (
            BitFieldLayout()
            .field("a", "u4")
            .field("b", "u4")
            .field("c", "u4")
            .compile()
        )
        plan = layout.write("k", values={"a": 1, "b": 99, "c": 3}, overflow="FAIL")
        assert plan.command == ["BITFIELD", "k", "SET", "u4", 0, 1, "SET", "u4", 8, 3]
        # the skipped middle field must not shift the values behind it
        assert plan.parse([7, 8]) == {"a": 7, "b": None, "c": 8}

    def test_fail_with_all_values_rejected(self):
        plan = make_layout().write(
            "k", values={"score": 99, "level": -1}, overflow="FAIL"
        )
        assert plan.operations == 0
        assert plan.skipped == ("score", "level")
        with pytest.raises(DataError, match="nothing to execute"):
            plan.command
        assert plan.parse([]) == {"score": None, "level": None}


class TestResponseParsing:
    def test_read_parse(self):
        plan = make_layout().read("k")
        assert plan.parse([-3, 9]) == {"score": -3, "level": 9}

    def test_parse_response_length_mismatch(self):
        layout = make_layout()
        with pytest.raises(InvalidResponse):
            layout.read("k").parse([1])
        with pytest.raises(InvalidResponse):
            layout.write("k", values={"score": 1}).parse([])

    def test_calls_do_not_pollute_each_other(self):
        layout = make_layout()
        values = {"score": -1, "level": 2047}
        plan1 = layout.write("k", record=0, values=values)
        plan2 = layout.write("k", record=1, values={"score": 0, "level": 1})
        # the caller's dict is only read, never mutated
        assert values == {"score": -1, "level": 2047}
        # plans are independent per-call objects
        plan1.command.append("junk")
        assert "junk" not in plan2.command
        result1 = plan1.parse([0, 0])
        result2 = plan2.parse([1, 2])
        result1["score"] = 12345
        assert result2 == {"score": 1, "level": 2}
        assert plan1.parse([0, 0]) == {"score": 0, "level": 0}
        # the shared compiled layout is unchanged
        assert layout.field_names == ("score", "level")
        read1 = layout.read("k", record=0)
        read2 = layout.read("k", record=1)
        assert read1.parse([1, 2]) is not read2.parse([3, 4])


class TestByteExactSemantics:
    def test_adjacent_records_and_exact_bytes(self):
        layout = make_layout()
        store = bytearray()

        plan = layout.write("k", record=0, values={"score": -1, "level": 2047})
        assert emulate_bitfield(store, plan.command) == [0, 0]
        # i5(-1)=11111, 3 gap bits, u11(2047)=11111111111, 5 stride pad bits
        assert bytes(store) == b"\xf8\xff\xe0"

        plan = layout.write("k", record=1, values={"score": -16, "level": 0})
        # SET returns the previous values of record 1 (all zero so far)
        assert emulate_bitfield(store, plan.command) == [0, 0]
        # writing record 1 must not clobber record 0
        assert bytes(store) == b"\xf8\xff\xe0\x80\x00\x00"

        plan = layout.read("k", record=0)
        assert plan.parse(emulate_bitfield(store, plan.command)) == {
            "score": -1,
            "level": 2047,
        }
        plan = layout.read("k", record=1)
        assert plan.parse(emulate_bitfield(store, plan.command)) == {
            "score": -16,
            "level": 0,
        }

    def test_cross_byte_boundary_field(self):
        # the u11 field starts at bit 8 and spans bytes 1 and 2
        layout = make_layout()
        store = bytearray()
        plan = layout.write("k", values={"level": 0b10101010101})
        emulate_bitfield(store, plan.command)
        assert bytes(store) == b"\x00\xaa\xa0"
        plan = layout.read("k", fields=["level"])
        assert plan.parse(emulate_bitfield(store, plan.command)) == {
            "level": 0b10101010101
        }

    def test_negative_and_extreme_values_round_trip(self):
        layout = make_layout()
        store = bytearray()
        cases = [
            ("score", -16),
            ("score", -1),
            ("score", 15),
            ("level", 0),
            ("level", 2047),
        ]
        for name, value in cases:
            plan = layout.write("k", values={name: value})
            emulate_bitfield(store, plan.command)
            plan = layout.read("k", fields=[name])
            assert plan.parse(emulate_bitfield(store, plan.command)) == {name: value}

    def test_overflow_modes_against_emulated_server(self):
        layout = make_layout()
        store = bytearray()

        plan = layout.write("k", values={"level": 2048}, overflow="WRAP")
        emulate_bitfield(store, plan.command)
        plan = layout.read("k", fields=["level"])
        assert plan.parse(emulate_bitfield(store, plan.command)) == {"level": 0}

        plan = layout.write("k", values={"level": 5000}, overflow="SAT")
        emulate_bitfield(store, plan.command)
        plan = layout.read("k", fields=["level"])
        assert plan.parse(emulate_bitfield(store, plan.command)) == {"level": 2047}

        plan = layout.write("k", values={"score": 7, "level": 5000}, overflow="FAIL")
        response = emulate_bitfield(store, plan.command)
        assert plan.parse(response) == {"score": 0, "level": None}
        plan = layout.read("k")
        assert plan.parse(emulate_bitfield(store, plan.command)) == {
            "score": 7,
            "level": 2047,  # untouched by the failed write
        }
