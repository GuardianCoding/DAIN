"""The benchmark record — INF-6.

Every number that reaches a slide comes out of infer/benchmarks/benchmarks.csv,
so this module freezes its shape. Liam builds against these column names while
the measurements are still being collected; if the schema moves after that his
slides break silently and nobody notices until Sunday.

The file is TIDY: one row per measurement, not one row per benchmark. The seven
INF-6 benchmarks have wildly different shapes — a boolean for "does it load", a
curve for fan-out, a pair for MTP on/off — and forcing them into one wide row
means a column set that changes every time you measure something new.

Filter by (benchmark_id, metric) for a number. By benchmark_id alone for a curve.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent / "benchmarks"
BENCH_PATH = BENCH_DIR / "benchmarks.csv"

# The seven numbers INF-6 requires, captured while things still work.
BENCHMARK_IDS: tuple[str, ...] = (
    "capacity",    # 1. loads on the cluster, fails on each node alone
    "ttft",        # 2. time to first token, 2k-token doc, 1 node vs all
    "decode",      # 3. decode tok/s per node and pooled
    "mtp",         # 4. speculative decoding on vs off
    "fanout",      # 5. wall clock at 1/2/3/4/5 nodes
    "placement",   # 6. Sean's split vs llama.cpp's default
    "recovery",    # 7. seconds from node death to serving again
)

# Units are part of the contract. "14.7" is not a number until you say tok/s.
UNITS: dict[str, str] = {
    "loads": "bool",
    "tok_s": "tok/s",
    "ttft_ms": "ms",
    "wall_clock_s": "s",
    "speedup": "x",
    "recovery_s": "s",
}

CSV_COLUMNS: tuple[str, ...] = (
    "benchmark_id",
    "model_id",
    "node_set",
    "n_nodes",
    "variant",
    "metric",
    "value",
    "unit",
    "repetitions",
    "recorded_at",
    "notes",
)


@dataclass(frozen=True)
class Measurement:
    """One measured number. Immutable — a recorded result is a fact."""

    benchmark_id: str
    model_id: str
    node_set: str          # "gpu-01" | "gpu-01+office-01" | "all"
    n_nodes: int
    variant: str           # "default" | "dain-placement" | "mtp-on" | "mtp-off" | "-"
    metric: str
    value: float
    unit: str = ""
    repetitions: int = 1
    recorded_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.benchmark_id not in BENCHMARK_IDS:
            raise ValueError(
                f"benchmark_id must be one of {', '.join(BENCHMARK_IDS)}; got {self.benchmark_id!r}"
            )
        if self.n_nodes < 1:
            raise ValueError(f"n_nodes must be at least 1, got {self.n_nodes}")
        if self.repetitions < 1:
            raise ValueError(f"repetitions must be at least 1, got {self.repetitions}")
        if not self.metric:
            raise ValueError("metric is required — a value with no metric is unreadable")

        # Frozen dataclasses need object.__setattr__ to fill derived defaults.
        if not self.recorded_at:
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            object.__setattr__(self, "recorded_at", stamp)
        if not self.unit:
            object.__setattr__(self, "unit", UNITS.get(self.metric, ""))


def append_measurement(measurement: Measurement, path: Path = BENCH_PATH) -> Path:
    """Append one row, writing the header if the file is new.

    Append rather than rewrite: measurements are collected across a whole
    weekend by several people, and a rewrite loses whatever it did not know.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(asdict(measurement))
    return path


def load_measurements(path: Path = BENCH_PATH) -> tuple[Measurement, ...]:
    """Read the record back, validating every row at the boundary."""
    if not path.is_file():
        raise FileNotFoundError(f"benchmark record not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
        return tuple(_row_to_measurement(row) for row in reader)


def _row_to_measurement(row: dict) -> Measurement:
    try:
        return Measurement(
            benchmark_id=row["benchmark_id"],
            model_id=row["model_id"],
            node_set=row["node_set"],
            n_nodes=int(row["n_nodes"]),
            variant=row["variant"],
            metric=row["metric"],
            value=float(row["value"]),
            unit=row["unit"],
            repetitions=int(row["repetitions"] or 1),
            recorded_at=row["recorded_at"],
            notes=row["notes"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"malformed benchmark row {row!r}: {error}") from error


def slide_number(
    measurements: tuple[Measurement, ...],
    benchmark_id: str,
    metric: str,
    *,
    variant: str | None = None,
    n_nodes: int | None = None,
) -> float:
    """Look up exactly one number for a slide. Raises rather than guessing.

    Liam's entry point. Ambiguity is an error on purpose: quietly returning the
    first of three matches is how a slide ends up with last night's figure.
    """
    matches = [
        item for item in measurements
        if item.benchmark_id == benchmark_id
        and item.metric == metric
        and (variant is None or item.variant == variant)
        and (n_nodes is None or item.n_nodes == n_nodes)
    ]
    if not matches:
        raise KeyError(f"no measurement for benchmark_id={benchmark_id!r} metric={metric!r}")
    if len(matches) > 1:
        detail = ", ".join(f"{item.variant}/{item.n_nodes}n" for item in matches)
        raise KeyError(
            f"{len(matches)} measurements match benchmark_id={benchmark_id!r} "
            f"metric={metric!r} ({detail}). Narrow with variant= or n_nodes=."
        )
    return matches[0].value


def missing_benchmarks(measurements: tuple[Measurement, ...]) -> tuple[str, ...]:
    """Which of the seven INF-6 numbers have not been captured yet."""
    recorded = {item.benchmark_id for item in measurements}
    return tuple(name for name in BENCHMARK_IDS if name not in recorded)
