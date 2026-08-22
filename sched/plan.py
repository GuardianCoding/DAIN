# sched/plan.py
"""
SCH-3: assign layers proportional to measured speed, then repair any node
that overflows its memory by pushing excess onto the fastest node with slack.

Pure function. No I/O, no state held between calls. Abdallah's control
plane calls plan(profiles, metrics, model_spec) directly — profiles from
REGISTRY.list_profiles(), metrics from REGISTRY.latest_metrics().
"""

from sched.cost import (
    fits,
    layer_weight_mb,
    overflow_mb,
    predict_tok_s,
    usable_mem_mb,
)

from contracts import Assignment, NodeMetrics, NodeProfile

"""
model_spec — the per-model data contract that sched.plan.plan() and
sched.cost.* need. This is NOT YET a real dataclass in contracts.py —
it's currently passed around as a plain dict. This file documents the
expected shape so integration doesn't require reverse-engineering it
from cost.py's internals.

WHO OWNS THIS DATA: Youssef (INF-3 model ladder, INF-4 MoE tuning).
These are facts about the model files themselves — layer count, size on
disk, KV cache footprint — not anything the scheduler measures or decides.

WHO BUILDS THE DICT: currently unresolved. Two options:
  (a) Abdallah's control plane resolves `model_id` -> this dict (e.g. via
      a models.toml lookup) BEFORE calling plan(), and plan() just
      receives the resolved dict.
  (b) plan() takes a bare `model_id: str` and does the models.toml
      lookup itself internally.
Option (a) keeps sched/ free of file I/O and config-parsing concerns,
which is why plan()'s current signature assumes it. CONFIRM THIS WITH
ABDALLAH before wiring real integration — the mock currently takes a
bare model_id string (MockControlPlane.plan(self, model_id: str)), which
matches option (b), so this may need to change on one side or the other.

--------------------------------------------------------------------------
REQUIRED SHAPE (dict[str, Any] for now; candidate for a real dataclass
once confirmed):

{
    "model_id": str,
        # Must match the model_id used elsewhere (Assignment.model_id,
        # the ?model= query param on GET /api/plan). Used purely as an
        # identifier / for error messages — not read for any calculation.
        # Example: "gpt-oss-120b"

    "total_layers": int,
        # L in the SCH-2 formula: t_token = Σ(layers_i / (L · tg_tok_s_i)) + hops·rtt
        # Total transformer layers in the model. This is a fixed
        # architectural fact of the model file, found in its config
        # (e.g. HF config.json's num_hidden_layers, or llama.cpp's
        # reported layer count via --list-devices / model metadata).
        # Example: 48

    "file_size_mb": int,
        # Total size of the GGUF file on disk, in MB. Used by
        # cost.layer_weight_mb() as file_size_mb / total_layers — a
        # UNIFORM approximation assuming every layer is the same size.
        # NOTE: this is an approximation, not exact for MoE models where
        # individual layers can have different active-expert footprints.
        # Good enough for the repair loop's purposes; flag to Youssef if
        # per-layer size data becomes available and this should be
        # replaced with a real per-layer breakdown instead.
        # Example: 63000  (gpt-oss-120b, ~63 GB per Table 2.3)

    "kv_mb_per_layer": float,
        # KV cache memory (MB) that ONE layer consumes at the context
        # length actually used in the demo. This is NOT a fixed model
        # fact alone — it depends on:
        #   - the model's architecture (hidden dim, num KV heads, head dim)
        #   - the context length you're planning for (§3.3: "KV cache is
        #     memory too... at long context it can exceed the model")
        # THIS IS THE FIELD MOST LIKELY TO BE MISSING/WRONG RIGHT NOW.
        # No test data has confirmed this number for any real model yet.
        # If unavailable, cost.fits()/overflow_mb() will systematically
        # underestimate memory footprint at long context — models that
        # "fit" on paper may still OOM in practice. Get this from
        # Youssef before trusting fits() near a node's memory ceiling.
}

--------------------------------------------------------------------------
WHAT READS THIS DICT:

  cost.layer_weight_mb(model_spec)
      -> model_spec["file_size_mb"] / model_spec["total_layers"]

  cost.node_footprint_mb(assignment, model_spec, node_id)
      -> uses total_layers (via layer_weight_mb) and kv_mb_per_layer

  cost.fits(assignment, profiles, metrics, model_spec)
      -> calls node_footprint_mb per node, needs both above

  cost.predict_tok_s(assignment, profiles, model_spec)
      -> uses model_spec["total_layers"] directly (the "L" in the formula)

  plan.plan(profiles, metrics, model_spec)
      -> passes model_spec through to all of the above, and reads
         model_spec["model_id"] once, for the Assignment.model_id field
         and for RuntimeError messages.

--------------------------------------------------------------------------
MINIMAL EXAMPLE (values are illustrative, not measured):

    model_spec = {
        "model_id": "gpt-oss-120b",
        "total_layers": 48,
        "file_size_mb": 63_000,
        "kv_mb_per_layer": 12.5,   # UNCONFIRMED — placeholder only
    }
"""


def plan(
    profiles: list[NodeProfile],
    metrics: list[NodeMetrics],
    model_spec: dict,
) -> Assignment:
    if not profiles:
        raise RuntimeError("no nodes are available")

    metrics_by_id = {m.node_id: m for m in metrics}
    # A node with a profile but no heartbeat yet can't be trusted for placement.
    usable_profiles = [
        profile
        for profile in profiles
        if profile.id in metrics_by_id and profile.state != "offline"
    ]
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
        predicted_tok_s=0.0,  # filled in by plan() after repair
        rationale="",  # filled in by plan() after repair
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

    max_iterations = model_spec["total_layers"] * len(
        profiles
    )  # hard stop, avoid infinite loop
    for _ in range(max_iterations):
        overflowing = [
            node_id
            for node_id in layers
            if overflow_mb(
                assignment,
                profiles,
                metrics,
                model_spec,
                node_id,
            )
            > 0
        ]
        if not overflowing:
            break

        node_id = overflowing[0]
        first, last = layers[node_id]
        node_count = last - first + 1
        if node_count <= 1:
            continue

        donor_id = _find_slack_donor(
            layers, profiles_by_id, metrics_by_id, model_spec, exclude=node_id
        )
        if donor_id is None:
            break

        # Shrink the overflowing node by one layer...
        layers[node_id] = (first, last - 1)
        # ...and grow the donor by one layer.
        donor_first, donor_last = layers[donor_id]
        layers[donor_id] = (donor_first, donor_last + 1)

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
    final_layers = _renumber_contiguous(layers, profiles)
    return Assignment(
        model_id=assignment.model_id,
        layers=final_layers,
        n_cpu_moe=assignment.n_cpu_moe,
        tensor_split=_tensor_split_from_layers(final_layers),
        predicted_tok_s=assignment.predicted_tok_s,
        rationale=assignment.rationale,
    )


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

        per_layer_mb = layer_weight_mb(model_spec) + model_spec["kv_mb_per_layer"]
        current_mb = (layers[node_id][1] - layers[node_id][0] + 1) * per_layer_mb
        slack_mb = usable_mem_mb(profile, metric) - current_mb
        if slack_mb >= per_layer_mb:
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
        result[node_id] = (
            (cursor, cursor + count - 1) if count > 0 else (cursor, cursor - 1)
        )
        cursor += count
    return result


def _tensor_split_from_layers(
    layers: dict[str, tuple[int, int]],
) -> list[float]:
    """Return layer shares in the planner's sorted node order."""
    ordered_ids = sorted(layers)
    counts = [
        max(0, layers[node_id][1] - layers[node_id][0] + 1) for node_id in ordered_ids
    ]

    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]

    return [round(count / total, 4) for count in counts]


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
