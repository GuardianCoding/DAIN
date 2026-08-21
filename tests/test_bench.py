"""Tests for infer/bench.py — the schema Liam builds his slides against.

The failure this guards: a slide showing a number that was never measured, or
the wrong one of three that matched. Both are silent, and both surface on
Sunday morning.
"""

from __future__ import annotations

import pytest

from infer.bench import (
    BENCH_PATH,
    BENCHMARK_IDS,
    CSV_COLUMNS,
    UNITS,
    Measurement,
    append_measurement,
    load_measurements,
    missing_benchmarks,
    slide_number,
)


def measurement(**overrides) -> Measurement:
    defaults = dict(
        benchmark_id="decode",
        model_id="working",
        node_set="gpu-01",
        n_nodes=1,
        variant="default",
        metric="tok_s",
        value=20.0,
    )
    return Measurement(**{**defaults, **overrides})


class TestMeasurement:
    def test_fills_the_unit_from_the_metric(self):
        # Arrange / Act
        result = measurement(metric="ttft_ms")

        # Assert — "14.7" is not a number until you say what it measures
        assert result.unit == UNITS["ttft_ms"]

    def test_keeps_an_explicit_unit(self):
        assert measurement(metric="tok_s", unit="tokens/second").unit == "tokens/second"

    def test_leaves_the_unit_blank_for_an_unknown_metric(self):
        assert measurement(metric="something_new").unit == ""

    def test_stamps_recorded_at_in_utc(self):
        assert measurement().recorded_at.endswith("+00:00")

    def test_keeps_an_explicit_timestamp(self):
        stamp = "2026-08-21T22:30:00+00:00"
        assert measurement(recorded_at=stamp).recorded_at == stamp

    def test_rejects_an_unknown_benchmark_id(self):
        # A typo'd id silently creates a benchmark nobody is looking for.
        with pytest.raises(ValueError, match="benchmark_id must be one of"):
            measurement(benchmark_id="troughput")

    def test_rejects_a_missing_metric(self):
        with pytest.raises(ValueError, match="metric is required"):
            measurement(metric="")

    @pytest.mark.parametrize("n_nodes", [0, -1])
    def test_rejects_impossible_node_counts(self, n_nodes):
        with pytest.raises(ValueError, match="n_nodes"):
            measurement(n_nodes=n_nodes)

    def test_rejects_zero_repetitions(self):
        with pytest.raises(ValueError, match="repetitions"):
            measurement(repetitions=0)


class TestRoundTrip:
    def test_writes_a_header_then_appends_without_repeating_it(self, tmp_path):
        # Arrange
        path = tmp_path / "benchmarks.csv"

        # Act
        append_measurement(measurement(), path)
        append_measurement(measurement(value=21.0), path)

        # Assert
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == ",".join(CSV_COLUMNS)
        assert len(lines) == 3

    def test_survives_a_round_trip(self, tmp_path):
        # Arrange
        path = tmp_path / "benchmarks.csv"
        original = measurement(value=14.7, variant="dain-placement", notes="office-02 one DIMM")

        # Act
        append_measurement(original, path)
        (restored,) = load_measurements(path)

        # Assert
        assert restored == original

    def test_appending_never_loses_earlier_rows(self, tmp_path):
        # Five people collect over a weekend; a rewrite loses what it did not know.
        path = tmp_path / "benchmarks.csv"
        for nodes in range(1, 6):
            append_measurement(measurement(benchmark_id="fanout", n_nodes=nodes), path)
        assert len(load_measurements(path)) == 5

    def test_rejects_a_file_missing_columns(self, tmp_path):
        # Arrange
        path = tmp_path / "benchmarks.csv"
        path.write_text("benchmark_id,value\ndecode,20.0\n", encoding="utf-8")

        # Act / Assert
        with pytest.raises(ValueError, match="missing column"):
            load_measurements(path)

    def test_rejects_a_malformed_row(self, tmp_path):
        # Arrange
        path = tmp_path / "benchmarks.csv"
        append_measurement(measurement(), path)
        corrupted = path.read_text(encoding="utf-8").replace(",20.0,", ",not-a-number,")
        path.write_text(corrupted, encoding="utf-8")

        # Act / Assert
        with pytest.raises(ValueError, match="malformed benchmark row"):
            load_measurements(path)

    def test_raises_when_the_record_is_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_measurements(tmp_path / "nope.csv")


class TestSlideNumber:
    @pytest.fixture
    def measurements(self):
        return (
            measurement(benchmark_id="placement", variant="default", value=8.2),
            measurement(benchmark_id="placement", variant="dain-placement", value=14.7),
            measurement(benchmark_id="fanout", metric="wall_clock_s", n_nodes=1, value=220.0),
            measurement(benchmark_id="fanout", metric="wall_clock_s", n_nodes=5, value=67.0),
        )

    def test_returns_the_single_match(self, measurements):
        assert slide_number(measurements, "placement", "tok_s", variant="dain-placement") == 14.7

    def test_narrows_by_node_count(self, measurements):
        assert slide_number(measurements, "fanout", "wall_clock_s", n_nodes=5) == 67.0

    def test_raises_rather_than_guessing_between_matches(self, measurements):
        # Returning the first of two is how a slide gets last night's figure.
        with pytest.raises(KeyError, match="2 measurements match"):
            slide_number(measurements, "placement", "tok_s")

    def test_raises_when_nothing_was_measured(self, measurements):
        with pytest.raises(KeyError, match="no measurement"):
            slide_number(measurements, "recovery", "recovery_s")


class TestCoverageOfInf6:
    def test_reports_everything_as_missing_when_nothing_is_recorded(self):
        assert missing_benchmarks(()) == BENCHMARK_IDS

    def test_reports_only_what_is_still_outstanding(self):
        recorded = (measurement(benchmark_id="decode"), measurement(benchmark_id="ttft"))
        outstanding = missing_benchmarks(recorded)
        assert "decode" not in outstanding and "ttft" not in outstanding
        assert "capacity" in outstanding

    def test_all_seven_inf6_benchmarks_are_defined(self):
        assert set(BENCHMARK_IDS) == {
            "capacity", "ttft", "decode", "mtp", "fanout", "placement", "recovery",
        }

    def test_committed_record_has_the_frozen_header(self):
        # Liam builds against these names. If this fails, his slides break.
        header = BENCH_PATH.read_text(encoding="utf-8").splitlines()[0]
        assert header == ",".join(CSV_COLUMNS)
