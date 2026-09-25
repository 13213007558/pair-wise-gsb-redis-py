import math

import pytest
from redis.commands.timeseries.commands import (
    DEFAULT_MAX_RESAMPLE_POINTS,
    TimeSeriesCommands,
)
from redis.commands.timeseries.utils import (
    RESAMPLE_EXACT,
    RESAMPLE_INTERPOLATED,
    RESAMPLE_LINEAR,
    RESAMPLE_MISSING,
    RESAMPLE_OBSERVED,
    RESAMPLE_PREVIOUS,
    build_resample_grid,
    build_resample_result,
    resample_series,
)
from redis.exceptions import DataError


class FakeTimeSeries(TimeSeriesCommands):
    """Sync TimeSeriesCommands stub returning a canned TS.MRANGE reply."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def execute_command(self, *args, **kwargs):
        self.calls.append(args)
        return self.response


class AsyncFakeTimeSeries(TimeSeriesCommands):
    """Async variant; execute_command is a coroutine."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    async def execute_command(self, *args, **kwargs):
        self.calls.append(args)
        return self.response


def _unified_response():
    # deliberately unordered: s2 arrives before s1
    return {
        b"s2": [{b"grp": b"x"}, {"aggregators": []}, [[0, 10.0], [100, 11.0]]],
        b"s1": [{b"grp": b"x"}, {"aggregators": []}, [[0, 1.0], [100, 3.0]]],
    }


class TestBuildResampleGrid:
    def test_basic(self):
        assert build_resample_grid(0, 10, 0, 30) == [0, 10, 20]

    def test_origin_offset(self):
        assert build_resample_grid(5, 10, 0, 30) == [5, 15, 25]

    def test_negative_timestamps_align(self):
        assert build_resample_grid(-1000, 250, -500, 0) == [-500, -250]
        assert build_resample_grid(0, 10, -25, 5) == [-20, -10, 0]

    def test_half_open_interval(self):
        assert build_resample_grid(0, 10, 0, 20) == [0, 10]
        assert build_resample_grid(0, 10, 3, 27) == [10, 20]

    def test_empty(self):
        assert build_resample_grid(0, 10, 1, 9) == []
        assert build_resample_grid(0, 10, 20, 10) == []


class TestResampleSeries:
    SAMPLES = [(0, 1.0), (100, 2.0), (300, 4.0)]

    def test_exact(self):
        values, mask = resample_series(self.SAMPLES, [0, 50, 100], RESAMPLE_EXACT)
        assert values == [1.0, None, 2.0]
        assert mask == [RESAMPLE_OBSERVED, RESAMPLE_MISSING, RESAMPLE_OBSERVED]

    def test_exact_ignores_max_gap(self):
        values, mask = resample_series(self.SAMPLES, [100], RESAMPLE_EXACT, max_gap=0)
        assert values == [2.0]
        assert mask == [RESAMPLE_OBSERVED]

    def test_exact_nan_is_observed(self):
        values, mask = resample_series([(0, float("nan"))], [0], RESAMPLE_EXACT)
        assert math.isnan(values[0])
        assert mask == [RESAMPLE_OBSERVED]

    def test_previous(self):
        values, mask = resample_series(self.SAMPLES, [50, 100, 150], RESAMPLE_PREVIOUS)
        assert values == [1.0, 2.0, 2.0]
        assert mask == [
            RESAMPLE_INTERPOLATED,
            RESAMPLE_OBSERVED,
            RESAMPLE_INTERPOLATED,
        ]

    def test_previous_max_gap(self):
        values, mask = resample_series(
            self.SAMPLES, [150, 200, 201], RESAMPLE_PREVIOUS, max_gap=100
        )
        # 150: 50ms from the sample at 100 -> ok
        # 200: exactly max_gap (100ms) from the sample at 100 -> ok
        # 201: 101ms from the sample at 100 -> beyond max_gap -> missing
        assert values == [2.0, 2.0, None]
        assert mask == [
            RESAMPLE_INTERPOLATED,
            RESAMPLE_INTERPOLATED,
            RESAMPLE_MISSING,
        ]

    def test_previous_before_first_sample(self):
        values, mask = resample_series(self.SAMPLES, [-1], RESAMPLE_PREVIOUS)
        assert values == [None]
        assert mask == [RESAMPLE_MISSING]

    def test_linear(self):
        values, mask = resample_series(self.SAMPLES, [50, 200], RESAMPLE_LINEAR)
        assert values == [1.5, 3.0]
        assert mask == [RESAMPLE_INTERPOLATED, RESAMPLE_INTERPOLATED]

    def test_linear_never_extrapolates(self):
        values, mask = resample_series(self.SAMPLES, [-50, 350], RESAMPLE_LINEAR)
        assert values == [None, None]
        assert mask == [RESAMPLE_MISSING, RESAMPLE_MISSING]

    def test_linear_max_gap(self):
        samples = [(0, 1.0), (1000, 3.0)]
        # right neighbor is 750ms away, beyond max_gap -> missing
        values, mask = resample_series(samples, [250], RESAMPLE_LINEAR, max_gap=500)
        assert values == [None]
        assert mask == [RESAMPLE_MISSING]
        # both neighbors exactly max_gap away -> interpolated
        values, mask = resample_series(samples, [500], RESAMPLE_LINEAR, max_gap=500)
        assert values == [2.0]
        assert mask == [RESAMPLE_INTERPOLATED]

    def test_linear_nan_neighbor_propagates(self):
        values, mask = resample_series(
            [(0, 1.0), (100, float("nan"))], [50], RESAMPLE_LINEAR
        )
        assert math.isnan(values[0])
        assert mask == [RESAMPLE_INTERPOLATED]

    def test_sample_exactly_on_grid_point_is_observed(self):
        values, mask = resample_series(self.SAMPLES, [300], RESAMPLE_LINEAR)
        assert values == [4.0]
        assert mask == [RESAMPLE_OBSERVED]


class TestBuildResampleResult:
    def test_columns_sorted_when_keys_not_given(self):
        result = build_resample_result(_unified_response(), [0, 100], RESAMPLE_EXACT)
        assert result["keys"] == [b"s1", b"s2"]
        assert result["labels"] == [{b"grp": b"x"}, {b"grp": b"x"}]
        assert result["values"] == [[1.0, 10.0], [3.0, 11.0]]
        assert result["mask"] == [
            [RESAMPLE_OBSERVED, RESAMPLE_OBSERVED],
            [RESAMPLE_OBSERVED, RESAMPLE_OBSERVED],
        ]

    def test_identical_labels_do_not_merge_columns(self):
        response = {
            b"a": [{b"grp": b"same"}, {"aggregators": []}, [[0, 1.0]]],
            b"b": [{b"grp": b"same"}, {"aggregators": []}, [[0, 2.0]]],
        }
        result = build_resample_result(response, [0], RESAMPLE_EXACT)
        assert result["keys"] == [b"a", b"b"]
        assert result["labels"] == [{b"grp": b"same"}, {b"grp": b"same"}]
        assert result["values"] == [[1.0, 2.0]]

    def test_legacy_list_shape(self):
        response = [
            {b"s2": [[], [(0, 10.0)]]},
            {b"s1": [[], [(0, 1.0)]]},
        ]
        result = build_resample_result(response, [0], RESAMPLE_EXACT)
        assert result["keys"] == [b"s1", b"s2"]
        assert result["values"] == [[1.0, 10.0]]

    def test_caller_fixed_column_order(self):
        result = build_resample_result(
            _unified_response(), [0], RESAMPLE_EXACT, keys=["s2", "s1", "s3"]
        )
        # discovered columns report the server-side key; the unknown "s3"
        # column echoes the caller-provided key
        assert result["keys"] == [b"s2", b"s1", "s3"]
        assert result["labels"] == [{b"grp": b"x"}, {b"grp": b"x"}, {}]
        assert result["values"] == [[10.0, 1.0, None]]
        assert result["mask"] == [
            [RESAMPLE_OBSERVED, RESAMPLE_OBSERVED, RESAMPLE_MISSING]
        ]

    def test_empty_response(self):
        result = build_resample_result({}, [0], RESAMPLE_EXACT)
        assert result["keys"] == []
        assert result["labels"] == []
        assert result["values"] == [[]]
        assert result["mask"] == [[]]


class TestResampleCommand:
    def test_mrange_params_exact(self):
        client = FakeTimeSeries({})
        client.resample(["grp=x"], origin=0, step=100, start=0, end=300)
        args = client.calls[0]
        assert args[0] == "TS.MRANGE"
        assert args[1] == 0
        assert args[2] == 200
        assert "WITHLABELS" in args
        assert "AGGREGATION" not in args
        assert list(args[-2:]) == ["FILTER", "grp=x"]

    def test_mrange_params_previous_extends_left(self):
        client = FakeTimeSeries({})
        client.resample(["grp=x"], 0, 100, 0, 300, strategy="previous", max_gap=50)
        args = client.calls[0]
        assert args[1] == -50
        assert args[2] == 200

    def test_mrange_params_linear_extends_both_sides(self):
        client = FakeTimeSeries({})
        client.resample(["grp=x"], 0, 100, 0, 300, strategy="linear", max_gap=50)
        args = client.calls[0]
        assert args[1] == -50
        assert args[2] == 250

    def test_mrange_params_unbounded_gap(self):
        client = FakeTimeSeries({})
        client.resample(["grp=x"], 0, 100, 0, 300, strategy="previous")
        args = client.calls[0]
        assert args[1] == "-"
        assert args[2] == 200
        client = FakeTimeSeries({})
        client.resample(["grp=x"], 0, 100, 0, 300, strategy="linear")
        args = client.calls[0]
        assert args[1] == "-"
        assert args[2] == "+"

    def test_mrange_params_negative_grid(self):
        client = FakeTimeSeries({})
        client.resample(
            ["grp=x"], -1000, 100, -300, -100, strategy="linear", max_gap=50
        )
        args = client.calls[0]
        assert args[1] == -350
        assert args[2] == -150

    def test_validation_errors(self):
        client = FakeTimeSeries({})
        with pytest.raises(DataError):
            client.resample(["grp=x"], 0, 0, 0, 10)
        with pytest.raises(DataError):
            client.resample(["grp=x"], 0, 10, 0, 10, strategy="bogus")
        with pytest.raises(DataError):
            client.resample(["grp=x"], 0, 10, 10, 0)
        with pytest.raises(DataError):
            client.resample(["grp=x"], 0, 10, 0, 10, max_gap=-1)
        with pytest.raises(DataError):
            client.resample(["grp=x"], 0, 10, 0, 10, max_points=0)

    def test_max_points_limit(self):
        client = FakeTimeSeries({})
        with pytest.raises(DataError):
            client.resample(["grp=x"], 0, 1, 0, 5, max_points=3)
        client.resample(["grp=x"], 0, 1, 0, 5, max_points=5)
        client.resample(["grp=x"], 0, 1, 0, 5, max_points=None)
        with pytest.raises(DataError):
            client.resample(["grp=x"], 0, 1, 0, DEFAULT_MAX_RESAMPLE_POINTS + 1)

    def test_empty_grid_skips_query(self):
        client = FakeTimeSeries({})
        result = client.resample(["grp=x"], 0, 10, 1, 9, keys=["a", "b"])
        assert client.calls == []
        assert result == {
            "timestamps": [],
            "keys": ["a", "b"],
            "labels": [{}, {}],
            "values": [],
            "mask": [],
        }

    def test_result_matrix(self):
        client = FakeTimeSeries(_unified_response())
        result = client.resample(
            ["grp=x"], 0, 50, 0, 100, strategy="linear", max_gap=100
        )
        assert result["timestamps"] == [0, 50]
        assert result["keys"] == [b"s1", b"s2"]
        assert result["values"] == [[1.0, 10.0], [2.0, 10.5]]
        assert result["mask"] == [
            [RESAMPLE_OBSERVED, RESAMPLE_OBSERVED],
            [RESAMPLE_INTERPOLATED, RESAMPLE_INTERPOLATED],
        ]

    async def test_async_client(self):
        client = AsyncFakeTimeSeries(_unified_response())
        result = await client.resample(["grp=x"], 0, 100, 0, 200, strategy="previous")
        assert result["timestamps"] == [0, 100]
        assert result["keys"] == [b"s1", b"s2"]
        assert result["values"] == [[1.0, 10.0], [3.0, 11.0]]
        assert result["mask"] == [
            [RESAMPLE_OBSERVED, RESAMPLE_OBSERVED],
            [RESAMPLE_OBSERVED, RESAMPLE_OBSERVED],
        ]
