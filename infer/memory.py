"""Memory arithmetic for the DAIN inference fabric.

This module exists to settle one question with numbers instead of adjectives:
**does the capacity claim survive contact with a judge?**

"No single machine here can hold this model" is the easiest thing on stage to
falsify, because anyone can ask what is in the big desktop. On this cluster
gpu-01 holds ~77 GiB by itself, so weights alone almost never prove the claim.
What does prove it is weights + KV cache + concurrency, which is what
`capacity_report` computes.

All memory values are MiB (binary), matching `free -m` and
contracts.NodeProfile.ram_total_mb. Download sizes stay in decimal GB and live
in infer/models.py. Never mix them.

Every node is Linux, so `free -m` is the only reading that matters — including
on gpu-02, where it reports the WSL2 VM's allocation rather than the host's
16 GB. That is the correct number to plan against, because it is all the node
can ever give a model.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from infer.models import ModelSpec

CLUSTER_PATH = Path(__file__).resolve().parent.parent / "cluster.toml"

# What the OS and its resident set take before llama.cpp allocates anything.
# Starting estimates only — replace each with a measured number from `free -m`
# on an idle box. The difference decides whether a node participates at all.
OS_RESERVE_MIB: dict[str, int] = {
    "linux-headless": 800,
    "linux-desktop": 2500,
    # WSL2 (gpu-02). Small, because this is NOT the Windows host's reserve:
    # inside WSL, /proc/meminfo reports the VM's allocation, so the host's cut
    # has already been taken before this table is consulted. What is left to
    # reserve is the VM's own minimal userland.
    #
    # The corollary is that this number is only correct if [wsl].memory_gb in
    # cluster.toml matches what .wslconfig actually granted. WSL2 defaults to
    # half of host RAM, so an unset .wslconfig makes gpu-02 silently smaller
    # than the fixture claims. scripts/inventory.sh checks for exactly that.
    "linux-wsl": 512,
}

# Driver + compositor working set that is never available to a model.
VRAM_RESERVE_MIB = 800

# llama.cpp compute buffers: activations, logits, batch scratch. Scales with
# batch and context, not model size. Measure from the "compute buffer size"
# lines in llama-server's startup log and replace this.
DEFAULT_OVERHEAD_MIB = 1600

BYTES_PER_MIB = 1024**2


@dataclass(frozen=True)
class NodeBudget:
    """What one node can actually give a model, after the OS takes its cut."""

    node_id: str
    ram_usable_mib: int
    vram_usable_mib: int
    os_class: str
    backend: str
    verified: bool

    @property
    def total_usable_mib(self) -> int:
        return self.ram_usable_mib + self.vram_usable_mib

    @property
    def total_usable_gib(self) -> float:
        return self.total_usable_mib / 1024


@dataclass(frozen=True)
class KVGeometry:
    """Everything needed to size a KV cache. `source` is not decoration.

    Get a measured value from llama-server's startup line:
        llama_kv_cache: KV self size = NNNN MiB
    and set source="measured". An estimated geometry must never back a claim
    you make out loud.
    """

    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2                      # f16. --cache-type-k q8_0 makes this 1.
    full_attention_layers: int | None = None  # None = every layer is full attention
    source: str = "estimated"

    def __post_init__(self) -> None:
        if min(self.layers, self.kv_heads, self.head_dim, self.dtype_bytes) <= 0:
            raise ValueError("KVGeometry fields must all be positive")
        if self.full_attention_layers is not None and not 0 < self.full_attention_layers <= self.layers:
            raise ValueError(
                f"full_attention_layers must be in 1..{self.layers}, got {self.full_attention_layers}"
            )

    @property
    def cached_layers(self) -> int:
        """Sliding-window layers hold a fixed tiny window, so they don't scale."""
        return self.full_attention_layers or self.layers


# ESTIMATED geometry for gpt-oss-120b, from its published architecture: 36
# blocks, 8 KV heads x 64 head dim, alternating full / sliding-window(128)
# attention, so ~18 layers hold a context-scaled cache at 2 KiB per token each.
# VERIFY against llama-server's startup log before this backs anything on stage.
GPT_OSS_120B_KV = KVGeometry(
    layers=36, kv_heads=8, head_dim=64, full_attention_layers=18, source="estimated"
)


@dataclass(frozen=True)
class Footprint:
    """Total memory one model configuration needs to load and serve."""

    model_id: str
    weights_mib: float
    kv_mib: float
    overhead_mib: float
    context_tokens: int
    slots: int

    @property
    def total_mib(self) -> float:
        return self.weights_mib + self.kv_mib + self.overhead_mib

    @property
    def total_gib(self) -> float:
        return self.total_mib / 1024


@dataclass(frozen=True)
class FitResult:
    fits: bool
    required_mib: float
    available_mib: float

    @property
    def headroom_mib(self) -> float:
        return self.available_mib - self.required_mib

    @property
    def headroom_gib(self) -> float:
        return self.headroom_mib / 1024


class Profile(Protocol):
    """Structurally identical to contracts.NodeProfile, imported by neither.

    This is the RUNTIME source of truth: what a live node reported at join.
    Nothing here reads a configured address, because a configured address
    outlives the node that owned it.
    """

    id: str
    ram_total_mb: int
    ram_free_mb: int
    vram_total_mb: int
    backend: str


def budget_from_profile(profile: Profile, os_class: str) -> NodeBudget:
    """Build a budget from a live node's self-report — the normal path.

    Prefers the node's MEASURED free memory over `total - OS_RESERVE_MIB`,
    because a real reading beats a table of guesses. The reserve table is only
    a fallback for planning runs where no node has reported yet.
    """
    if os_class not in OS_RESERVE_MIB:
        raise ValueError(
            f"{profile.id}: os_class must be one of "
            f"{', '.join(sorted(OS_RESERVE_MIB))}; got {os_class!r}"
        )
    if profile.ram_total_mb <= 0:
        raise ValueError(f"{profile.id}: ram_total_mb must be positive, got {profile.ram_total_mb}")

    measured = profile.ram_free_mb
    ram_usable = measured if measured > 0 else max(0, profile.ram_total_mb - OS_RESERVE_MIB[os_class])
    return NodeBudget(
        node_id=profile.id,
        ram_usable_mib=ram_usable,
        vram_usable_mib=max(0, profile.vram_total_mb - VRAM_RESERVE_MIB) if profile.vram_total_mb else 0,
        os_class=os_class,
        backend=profile.backend,
        verified=measured > 0,   # a live measurement is what "verified" means
    )


def planning_budgets(path: Path = CLUSTER_PATH) -> tuple[NodeBudget, ...]:
    """Read the OFFLINE planning fixture from cluster.toml [[planning.nodes]].

    For answering "which models are worth downloading?" before any node exists.
    Never used at runtime — the live path is budget_from_profile.
    """
    if not path.is_file():
        raise FileNotFoundError(f"cluster file not found: {path}")

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    nodes = raw.get("planning", {}).get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"{path} has no [[planning.nodes]] entries")

    return tuple(_parse_budget(body) for body in nodes)


def _parse_budget(body: dict) -> NodeBudget:
    """Validate one planning entry and subtract the OS reserve."""
    node_id = str(body.get("id", "")).strip()
    if not node_id:
        raise ValueError("every [[planning.nodes]] entry needs an id")

    os_class = str(body.get("os_class", ""))
    if os_class not in OS_RESERVE_MIB:
        raise ValueError(
            f"planning node {node_id}: os_class must be one of "
            f"{', '.join(sorted(OS_RESERVE_MIB))}; got {os_class!r}"
        )

    ram_mb = body.get("ram_mb")
    if not isinstance(ram_mb, int) or ram_mb <= 0:
        raise ValueError(f"planning node {node_id}: ram_mb must be a positive int, got {ram_mb!r}")

    vram_mb = body.get("vram_mb", 0)
    if not isinstance(vram_mb, int) or vram_mb < 0:
        raise ValueError(f"planning node {node_id}: vram_mb must be a non-negative int, got {vram_mb!r}")

    return NodeBudget(
        node_id=node_id,
        ram_usable_mib=max(0, ram_mb - OS_RESERVE_MIB[os_class]),
        vram_usable_mib=max(0, vram_mb - VRAM_RESERVE_MIB) if vram_mb else 0,
        os_class=os_class,
        backend=str(body.get("backend", "cpu")),
        verified=bool(body.get("verified", False)),
    )


def kv_cache_mib(geometry: KVGeometry, context_tokens: int, slots: int = 1) -> float:
    """KV cache for `slots` concurrent sequences of `context_tokens` each.

    Concurrency multiplies this, and that is the whole argument for the headline
    model: one agent session fits on gpu-01, four do not.
    """
    if context_tokens <= 0 or slots <= 0:
        raise ValueError(f"context_tokens and slots must be positive, got {context_tokens}, {slots}")

    bytes_per_token_per_layer = 2 * geometry.kv_heads * geometry.head_dim * geometry.dtype_bytes
    total_bytes = bytes_per_token_per_layer * geometry.cached_layers * context_tokens * slots
    return total_bytes / BYTES_PER_MIB


def footprint(
    spec: ModelSpec,
    geometry: KVGeometry,
    context_tokens: int,
    slots: int = 1,
    overhead_mib: float = DEFAULT_OVERHEAD_MIB,
) -> Footprint:
    """Total memory needed to load `spec` at this context and concurrency."""
    return Footprint(
        model_id=spec.model_id,
        weights_mib=spec.weights_mib,
        kv_mib=kv_cache_mib(geometry, context_tokens, slots),
        overhead_mib=overhead_mib,
        context_tokens=context_tokens,
        slots=slots,
    )


def fits(budget: NodeBudget, need: Footprint) -> FitResult:
    return FitResult(
        fits=budget.total_usable_mib >= need.total_mib,
        required_mib=need.total_mib,
        available_mib=float(budget.total_usable_mib),
    )


def fits_pooled(budgets: tuple[NodeBudget, ...], need: Footprint) -> FitResult:
    pooled = float(sum(budget.total_usable_mib for budget in budgets))
    return FitResult(fits=pooled >= need.total_mib, required_mib=need.total_mib, available_mib=pooled)


def capacity_report(need: Footprint, budgets: tuple[NodeBudget, ...], geometry: KVGeometry) -> str:
    """INF-6 benchmark #1: does 'fails alone, loads pooled' actually hold?

    Prints the evidence, not just the verdict, so anyone on the team can check
    the claim before it reaches a projector.
    """
    if not budgets:
        raise ValueError("no compute nodes to report on")

    unverified = [budget.node_id for budget in budgets if not budget.verified]
    single = {budget.node_id: fits(budget, need) for budget in budgets}
    pooled = fits_pooled(budgets, need)
    holders = [node_id for node_id, result in single.items() if result.fits]

    lines = [
        f"CAPACITY REPORT — {need.model_id} @ {need.context_tokens:,} ctx x {need.slots} slot(s)",
        f"  weights {need.weights_mib / 1024:7.1f} GiB"
        f" | KV {need.kv_mib / 1024:6.1f} GiB ({geometry.source})"
        f" | overhead {need.overhead_mib / 1024:4.1f} GiB"
        f" | TOTAL {need.total_gib:7.1f} GiB",
        "",
        f"  {'NODE':<12} {'USABLE':>10} {'HEADROOM':>10}   ALONE",
    ]
    for budget in budgets:
        result = single[budget.node_id]
        lines.append(
            f"  {budget.node_id:<12} {budget.total_usable_gib:>8.1f}G "
            f"{result.headroom_gib:>9.1f}G   {'HOLDS' if result.fits else 'fails'}"
        )
    lines += [
        "",
        f"  {'POOLED':<12} {pooled.available_mib / 1024:>8.1f}G "
        f"{pooled.headroom_gib:>9.1f}G   {'LOADS' if pooled.fits else 'FAILS'}",
        "",
    ]

    if not pooled.fits:
        lines.append("  VERDICT: does not fit the cluster. Drop context, slots, or model.")
    elif holders:
        lines.append(
            f"  VERDICT: NOT a capacity claim — {', '.join(holders)} holds this alone. "
            f"Raise slots or context until no node does, or claim something else."
        )
    else:
        lines.append("  VERDICT: capacity claim HOLDS — fails on every node alone, loads pooled.")

    if unverified:
        lines.append(
            f"  WARNING: unverified node(s): {', '.join(unverified)}. "
            f"Run the inventory and set verified = true before saying this out loud."
        )
    if geometry.source != "measured":
        lines.append(
            "  WARNING: KV geometry is estimated. Confirm against llama-server's "
            "'KV self size' line before this backs anything on stage."
        )
    return "\n".join(lines)
