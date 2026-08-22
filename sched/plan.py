# sched/plan.py
"""
SCH-3: assign layers proportional to measured speed, then repair any node
that overflows its memory by pushing excess onto the fastest node with slack.

Pure function. No I/O, no state held between calls. Abdallah's control
plane calls plan(profiles, metrics, model_spec) directly — profiles from
REGISTRY.list_profiles(), metrics from REGISTRY.latest_metrics().
"""
from contracts import NodeProfile, NodeMetrics, Assignment
from cost import (
    usable_mem_mb,
    layer_weight_mb,
    node_footprint_mb,
    overflow_mb,
    fits,
    predict_tok_s,
)

def plan(
    profiles: list[NodeProfile],
    metrics: list[NodeMetrics],
    model_spec: dict,
) -> Assignment:
    if not profiles:
        raise RuntimeError("no nodes are available")

    metrics_by_id = {m.node_id: m for m in metrics}
    # A node with a profile but no heartbeat yet can't be trusted for placement.
    usable_profiles = [p for p in profiles if p.id in metrics_by_id]
    if not usable_profiles:
        raise RuntimeError("no nodes have reported metrics yet")

    assignment = _assign_by_speed(usable_profiles, model_spec)
    assignment = _repair(assignment, usable_profiles, metrics, model_spec)

    if not fits(assignment, usable_profiles, metrics, model_spec):
        raise RuntimeError(
            f"model {model_spec['model_id']} does not fit in pooled memory "
            f"even after repair"
        )

    assignment.predicted_tok_s = predict_tok_s(assignment, usable_profiles, model_spec)
    assignment.rationale = _explain(assignment, usable_profiles, metrics)
    return assignment


def _assign_by_speed(profiles: list[NodeProfile], model_spec: dict) -> Assignment:
    """Give each node a share of layers proportional to its measured tg_tok_s.
    Ignores memory entirely — that's what _repair fixes afterward."""
    total_layers = model_spec["total_layers"]
    total_speed = sum(p.tg_tok_s for p in profiles)

    if total_speed <= 0:
        raise RuntimeError(
            "no node has a measured tg_tok_s — profile the cluster before planning"
        )

    ordered = sorted(profiles, key=lambda p: p.id)  # stable, deterministic order
    layers: dict[str, tuple[int, int]] = {}
    first = 0
    allocated = 0

    for i, profile in enumerate(ordered):
        if i == len(ordered) - 1:
            count = total_layers - allocated  # last node takes the remainder
        else:
            share = profile.tg_tok_s / total_speed
            count = max(1, round(share * total_layers))
        count = min(count, total_layers - allocated)  # never overshoot
        last = first + count - 1
        layers[profile.id] = (first, last)
        first = last + 1
        allocated += count

    return Assignment(
        model_id=model_spec["model_id"],
        layers=layers,
        n_cpu_moe={p.id: 0 for p in profiles},  # Youssef's tuning fills this in later
        tensor_split=[round(p.tg_tok_s / total_speed, 4) for p in ordered],
        predicted_tok_s=0.0,   # filled in by plan() after repair
        rationale="",          # filled in by plan() after repair
    )


def _repair(
    assignment: Assignment,
    profiles: list[NodeProfile],
    metrics: list[NodeMetrics],
    model_spec: dict,
) -> Assignment:
    """Any node over its memory budget pushes layers onto the fastest node
    with slack. This loop is the actual algorithm, not the proportional
    split above — see §2.6."""
    profiles_by_id = {p.id: p for p in profiles}
    metrics_by_id = {m.node_id: m for m in metrics}
    layers = dict(assignment.layers)

    max_iterations = model_spec["total_layers"] * len(profiles)  # hard stop, avoid infinite loop
    for _ in range(max_iterations):
        overflowing = [
            node_id for node_id in layers
            if overflow_mb(assignment, profiles, metrics, model_spec, node_id) > 0
        ]
        if not overflowing:
            break

        node_id = overflowing[0]
        first, last = layers[node_id]
        node_count = last - first + 1
        if node_count <= 1:
            # Can't shrink further — nothing left to peel off this node.
            continue

        # Find the node with the most memory slack to absorb one layer.
        donor_id = _find_slack_donor(layers, profiles_by_id, metrics_by_id, model_spec, exclude=node_id)
        if donor_id is None:
            break  # no node has room; fits() check in plan() will catch this

        # Peel the last layer off the overflowing node onto the donor.
        layers[node_id] = (first, last - 1)
        donor_first, donor_last = layers[donor_id]
        # Keep layer ranges contiguous is NOT required here — donor gets an
        # extra layer count, actual layer *numbers* get renumbered by
        # _renumber_contiguous below so llama.cpp's --tensor-split still
        # sees a valid contiguous pipeline.
        assignment = Assignment(
            model_id=assignment.model_id,
            layers=_renumber_contiguous(layers, profiles),
            n_cpu_moe=assignment.n_cpu_moe,
            tensor_split=assignment.tensor_split,
            predicted_tok_s=assignment.predicted_tok_s,
            rationale=assignment.rationale,
        )
        layers = dict(assignment.layers)

    return assignment


def _find_slack_donor(
    layers: dict[str, tuple[int, int]],
    profiles_by_id: dict[str, NodeProfile],
    metrics_by_id: dict[str, NodeMetrics],
    model_spec: dict,
    exclude: str,
) -> str | None:
    """Fastest node with free memory headroom, excluding the overflowing node."""
    candidates = []
    for node_id in layers:
        if node_id == exclude:
            continue
        profile = profiles_by_id[node_id]
        metric = metrics_by_id[node_id]
        current_mb = (layers[node_id][1] - layers[node_id][0] + 1) * layer_weight_mb(model_spec)
        slack_mb = usable_mem_mb(profile, metric) - current_mb
        if slack_mb > layer_weight_mb(model_spec):  # room for at least one more layer
            candidates.append((profile.tg_tok_s, node_id))

    if not candidates:
        return None
    candidates.sort(reverse=True)  # fastest first
    return candidates[0][1]


def _renumber_contiguous(
    layers: dict[str, tuple[int, int]],
    profiles: list[NodeProfile],
) -> dict[str, tuple[int, int]]:
    """After moving layer counts between nodes, reassign actual layer
    numbers 0..L-1 contiguously in a fixed node order so the pipeline
    is still valid for --rpc / --tensor-split."""
    ordered_ids = sorted(layers.keys())
    counts = {}
    for i, node_id in enumerate(ordered_ids):
        first, last = layers[node_id]
        counts[node_id] = last - first + 1

    result = {}
    cursor = 0
    for node_id in ordered_ids:
        count = max(0, counts[node_id])
        result[node_id] = (cursor, cursor + count - 1) if count > 0 else (cursor, cursor - 1)
        cursor += count
    return result


def _explain(
    assignment: Assignment,
    profiles: list[NodeProfile],
    metrics: list[NodeMetrics],
) -> str:
    """SCH-5: plain English, goes on the dashboard."""
    profiles_by_id = {p.id: p for p in profiles}
    parts = []
    for node_id, (first, last) in sorted(assignment.layers.items()):
        count = last - first + 1
        bw = profiles_by_id[node_id].mem_bandwidth_gbs
        parts.append(f"{node_id} gets {count} layers ({bw:.0f} GB/s bandwidth)")
    return "; ".join(parts)