"""Load and validate the DAIN model ladder (infer/models.toml).

Units, stated once because mixing them is how a capacity claim dies on stage:
  * `size_gb` in models.toml is a DOWNLOAD size in decimal GB (what Hugging Face
    reports on a file listing).
  * Every `*_mib` value in this package is binary MiB (what `free -m` and Task
    Manager report). infer/memory.py works exclusively in MiB.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

BYTES_PER_GB = 1000**3
BYTES_PER_GIB = 1024**3
BYTES_PER_MIB = 1024**2

SLOW_LINK_MBPS = 20.0
FAST_LINK_MBPS = 200.0

LADDER_PATH = Path(__file__).resolve().parent / "models.toml"

REQUIRED_KEYS = ("role", "repo", "include", "size_gb", "priority")


@dataclass(frozen=True)
class ModelSpec:
    """One rung of the ladder. Immutable — the ladder is a fact, not state."""

    model_id: str
    role: str
    repo: str
    include: str
    size_gb: float
    params_total_b: float
    params_active_b: float
    priority: int
    claim: str
    notes: str

    @property
    def weights_mib(self) -> float:
        """Weights as they occupy memory, in MiB. Not the same number as size_gb."""
        return self.size_gb * BYTES_PER_GB / BYTES_PER_MIB

    @property
    def weights_gib(self) -> float:
        return self.size_gb * BYTES_PER_GB / BYTES_PER_GIB

    @property
    def sparsity(self) -> float:
        """Total params / active params. The number that predicts decode speed."""
        if self.params_active_b <= 0:
            raise ValueError(f"{self.model_id}: params_active_b must be positive")
        return self.params_total_b / self.params_active_b


def load_ladder(path: Path = LADDER_PATH) -> tuple[ModelSpec, ...]:
    """Parse and validate models.toml, sorted by priority. Fails loudly."""
    if not path.is_file():
        raise FileNotFoundError(f"model ladder not found: {path}")

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{path} has no [models.*] tables")

    specs = tuple(_parse_spec(model_id, body) for model_id, body in models.items())
    _reject_duplicate_priorities(specs)
    return tuple(sorted(specs, key=lambda spec: spec.priority))


def _parse_spec(model_id: str, body: object) -> ModelSpec:
    """Validate one [models.<id>] table at the boundary. Never trust the file."""
    if not isinstance(body, dict):
        raise ValueError(f"models.{model_id} is not a table")

    missing = [key for key in REQUIRED_KEYS if key not in body]
    if missing:
        raise ValueError(f"models.{model_id} missing required key(s): {', '.join(missing)}")

    size_gb = body["size_gb"]
    if not isinstance(size_gb, (int, float)) or size_gb <= 0:
        raise ValueError(f"models.{model_id}.size_gb must be positive, got {size_gb!r}")

    priority = body["priority"]
    if not isinstance(priority, int) or priority < 1:
        raise ValueError(f"models.{model_id}.priority must be an int >= 1, got {priority!r}")

    return ModelSpec(
        model_id=model_id,
        role=str(body["role"]),
        repo=str(body["repo"]),
        include=str(body["include"]),
        size_gb=float(size_gb),
        params_total_b=float(body.get("params_total_b", 0.0)),
        params_active_b=float(body.get("params_active_b", 0.0)),
        priority=priority,
        claim=str(body.get("claim", "")).strip(),
        notes=str(body.get("notes", "")).strip(),
    )


def _reject_duplicate_priorities(specs: tuple[ModelSpec, ...]) -> None:
    seen: dict[int, str] = {}
    for spec in specs:
        if spec.priority in seen:
            raise ValueError(
                f"duplicate priority {spec.priority}: "
                f"models.{seen[spec.priority]} and models.{spec.model_id}"
            )
        seen[spec.priority] = spec.model_id


def by_role(specs: tuple[ModelSpec, ...], role: str) -> ModelSpec:
    """Look up the single model filling a role. Raises if absent or ambiguous."""
    matches = [spec for spec in specs if spec.role == role]
    if not matches:
        known = ", ".join(sorted({spec.role for spec in specs}))
        raise KeyError(f"no model with role {role!r}. Known roles: {known}")
    if len(matches) > 1:
        names = ", ".join(spec.model_id for spec in matches)
        raise KeyError(f"role {role!r} is ambiguous across: {names}")
    return matches[0]


def estimate_hours(size_gb: float, mbps: float) -> float:
    """Wall-clock download estimate. `mbps` is megabits/s — what a speed test gives."""
    if mbps <= 0:
        raise ValueError(f"link speed must be positive, got {mbps}")
    return (size_gb * BYTES_PER_GB * 8) / (mbps * 1_000_000) / 3600


def format_ladder(specs: tuple[ModelSpec, ...]) -> str:
    """Ladder with cumulative size, so you can see where to stop if the link is bad."""
    header = (
        f"{'PRI':>3}  {'MODEL':<14} {'ROLE':<18} {'GB':>6} {'CUM':>7} "
        f"{'@20Mb':>7} {'@200Mb':>7}  REPO"
    )
    lines = [header, "-" * len(header)]
    cumulative = 0.0
    for spec in specs:
        cumulative += spec.size_gb
        lines.append(
            f"{spec.priority:>3}  {spec.model_id:<14} {spec.role:<18} "
            f"{spec.size_gb:>6.1f} {cumulative:>7.1f} "
            f"{estimate_hours(cumulative, SLOW_LINK_MBPS):>6.1f}h "
            f"{estimate_hours(cumulative, FAST_LINK_MBPS):>6.1f}h  {spec.repo}"
        )
    lines.append("-" * len(header))
    lines.append(f"{'':>3}  {'TOTAL':<14} {'':<18} {cumulative:>6.1f}")
    return "\n".join(lines)
