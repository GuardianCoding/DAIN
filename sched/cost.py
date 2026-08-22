# sched/cost.py
"""
Pure math only — no network, no state, nothing async. plan.py calls into
this repeatedly during assign-then-repair; you also call these functions
directly to validate SCH-2 ("predicted vs actual agrees within 20%").

model_spec shape assumed here — CONFIRM WITH YOUSSEF, this is his territory
(INF-3/INF-4) and the cost model is only as good as these numbers:
{
    "model_id": str,
    "total_layers": int,        # L
    "file_size_mb": int,        # total weight size on disk
    "kv_mb_per_layer": float,   # KV cache size per layer at the demo's context length
}
"""

from contracts import Assignment, NodeMetrics, NodeProfile

UNIFIED_MEMORY_BACKENDS = {"metal"}
COMPUTE_OVERHEAD_MB = 1600


def usable_mem_mb(profile: NodeProfile, metric: NodeMetrics) -> float:
    """Free memory this node can put layers into RIGHT NOW.
    Metal (unified memory) never adds vram_free_mb to ram_free_mb —
    they're the same physical pool. See mac-01 in the mock."""
    if profile.backend in UNIFIED_MEMORY_BACKENDS:
        base = metric.ram_free_mb
    else:
        base = metric.ram_free_mb + (metric.vram_free_mb or 0)
    return max(0.0, base - COMPUTE_OVERHEAD_MB)


def layer_weight_mb(model_spec: dict) -> float:
    """Uniform-layer approximation: total file size / total layers.
    Good enough for a repair loop; not exact for MoE models where
    individual layers can vary in expert count."""
    return model_spec["file_size_mb"] / model_spec["total_layers"]


def node_layer_count(assignment: Assignment, node_id: str) -> int:
    first, last = assignment.layers[node_id]
    return last - first + 1


def node_footprint_mb(assignment: Assignment, model_spec: dict, node_id: str) -> float:
    """What this node must hold: its share of weights + its share of KV cache.
    n_cpu_moe doesn't factor in here — that's WHERE within a node the expert
    weights sit (GPU vs RAM), an intra-node decision. This function is about
    cross-node placement, a layer either lives on this node or it doesn't."""
    layers = node_layer_count(assignment, node_id)
    weight_mb = layers * layer_weight_mb(model_spec)
    kv_mb = layers * model_spec["kv_mb_per_layer"]
    return weight_mb + kv_mb


def fits(
    assignment: Assignment,
    profiles: list[NodeProfile],
    metrics: list[NodeMetrics],
    model_spec: dict,
) -> bool:
    """SCH-2's constraint, checked per node:
    bytes(layers_i) + kv_share_i <= usable_mem_i"""
    metrics_by_id = {m.node_id: m for m in metrics}
    profiles_by_id = {p.id: p for p in profiles}

    for node_id in assignment.layers:
        profile = profiles_by_id[node_id]
        metric = metrics_by_id[node_id]
        footprint = node_footprint_mb(assignment, model_spec, node_id)
        if footprint > usable_mem_mb(profile, metric):
            return False
    return True


def overflow_mb(
    assignment: Assignment,
    profiles: list[NodeProfile],
    metrics: list[NodeMetrics],
    model_spec: dict,
    node_id: str,
) -> float:
    """How far OVER budget a single node is. 0 or negative = fits.
    Repair uses this to decide how many layers to peel off."""
    profile = next(p for p in profiles if p.id == node_id)
    metric = next(m for m in metrics if m.node_id == node_id)
    footprint = node_footprint_mb(assignment, model_spec, node_id)
    return footprint - usable_mem_mb(profile, metric)


def predict_tok_s(
    assignment: Assignment,
    profiles: list[NodeProfile],
    model_spec: dict,
) -> float:
    """t_token = sum(layers_i / (L * tg_tok_s_i)) + hops * rtt
    Returns 1 / t_token — predicted pipeline decode speed."""
    profiles_by_id = {p.id: p for p in profiles}
    L = model_spec["total_layers"]

    compute_time_s = 0.0
    for node_id, (first, last) in assignment.layers.items():
        layers_i = last - first + 1
        profile = profiles_by_id[node_id]
        if profile.tg_tok_s <= 0:
            raise ValueError(f"{node_id} has no measured tg_tok_s — profile it first")
        compute_time_s += layers_i / (L * profile.tg_tok_s)

    hops = max(0, len(assignment.layers) - 1)
    avg_rtt_s = (sum(p.rtt_ms for p in profiles) / len(profiles)) / 1000
    network_time_s = hops * avg_rtt_s

    t_token = compute_time_s + network_time_s
    if t_token <= 0:
        raise ValueError("t_token computed as zero — check profile/model_spec inputs")
    return 1 / t_token
