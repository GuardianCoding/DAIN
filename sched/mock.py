# sched/mock.py
"""
Example integration script for sched/. Mirrors the style of ctl/mock.py —
hardcoded, realistic NodeProfile/NodeMetrics data, run through cost.py and
plan.py so Abdallah (or anyone else) can see exactly how the pieces fit
together without needing a live cluster or the registry running.

Run directly: python -m sched.mock_example
Also importable: MOCK_PROFILES / MOCK_METRICS / MOCK_MODEL_SPEC are reused
by test_plan.py's fixtures, so the "example" data and the "test" data stay
in sync as one source of truth.
"""

from sched.cost import fits, node_footprint_mb
from sched.plan import plan

from contracts import NodeMetrics, NodeProfile

# ---------------------------------------------------------------------
# Hardcoded cluster — same four nodes as ctl/mock.py, but with realistic
# MEASURED values filled in (mock.py leaves these at 0.0 pending real
# profiling; here we fake plausible numbers so plan() has something to
# actually work with).
# ---------------------------------------------------------------------

MOCK_PROFILES = [
    NodeProfile(
        id="gpu-01",
        host="192.168.50.10",
        cpu="Intel Core i7",
        cores=12,
        ram_total_mb=64 * 1024,
        ram_free_mb=46 * 1024,
        gpu="NVIDIA GeForce RTX 5070 Ti",
        vram_total_mb=16 * 1024,
        backend="cuda",
        mem_bandwidth_gbs=650.0,
        tg_tok_s=45.0,
        pp_tok_s=180.0,
        rtt_ms=0.3,
        state="idle",
    ),
    NodeProfile(
        id="office-01",
        host="192.168.50.11",
        cpu="Intel Core i7 vPro",
        cores=8,
        ram_total_mb=8 * 1024,
        ram_free_mb=6 * 1024,
        gpu="Intel integrated graphics",
        vram_total_mb=0,
        backend="cpu",
        mem_bandwidth_gbs=38.0,
        tg_tok_s=9.0,
        pp_tok_s=65.0,
        rtt_ms=0.4,
        state="idle",
    ),
    NodeProfile(
        id="office-02",
        host="192.168.50.12",
        cpu="Intel Core i7 vPro",
        cores=8,
        ram_total_mb=8 * 1024,
        ram_free_mb=6 * 1024,
        gpu="Intel integrated graphics",
        vram_total_mb=0,
        backend="cpu",
        mem_bandwidth_gbs=19.0,  # single DIMM — half the bandwidth
        tg_tok_s=4.5,
        pp_tok_s=33.0,
        rtt_ms=0.4,
        state="idle",
    ),
    NodeProfile(
        id="mac-01",
        host="192.168.50.13",
        cpu="Apple M5 Pro",
        cores=12,
        ram_total_mb=24 * 1024,
        ram_free_mb=18 * 1024,
        gpu="Apple M5 Pro integrated GPU",
        vram_total_mb=24 * 1024,
        backend="metal",
        mem_bandwidth_gbs=270.0,
        tg_tok_s=22.0,
        pp_tok_s=110.0,
        rtt_ms=0.5,
        state="idle",
    ),
]

MOCK_METRICS = [
    NodeMetrics(
        node_id="gpu-01",
        timestamp=0.0,
        cpu_percent=18.0,
        ram_free_mb=48 * 1024,
        gpu_percent=25.0,
        vram_free_mb=12 * 1024,
        jobs_running=0,
    ),
    NodeMetrics(
        node_id="office-01",
        timestamp=0.0,
        cpu_percent=22.0,
        ram_free_mb=6 * 1024,
        gpu_percent=None,
        vram_free_mb=None,
        jobs_running=0,
    ),
    NodeMetrics(
        node_id="office-02",
        timestamp=0.0,
        cpu_percent=20.0,
        ram_free_mb=6 * 1024,
        gpu_percent=None,
        vram_free_mb=None,
        jobs_running=0,
    ),
    NodeMetrics(
        node_id="mac-01",
        timestamp=0.0,
        cpu_percent=15.0,
        ram_free_mb=14 * 1024,
        gpu_percent=10.0,
        vram_free_mb=None,
        jobs_running=0,
    ),
]

# STAND-IN per sched/model_spec.py — swap for real Youssef-supplied values
# the moment they're confirmed.
MOCK_MODEL_SPEC = {
    "model_id": "qwen3.6-35b-a3b",
    "total_layers": 40,
    "file_size_mb": 20 * 1024,  # ~20 GB, matches §3.3's "working model"
    "kv_mb_per_layer": 8.0,  # UNCONFIRMED placeholder
}


# ---------------------------------------------------------------------
# The actual walkthrough — what happens when you call plan()
# ---------------------------------------------------------------------


def run_example() -> None:
    print(f"Planning for model: {MOCK_MODEL_SPEC['model_id']}")
    print(f"Nodes available: {[p.id for p in MOCK_PROFILES]}\n")

    assignment = plan(MOCK_PROFILES, MOCK_METRICS, MOCK_MODEL_SPEC)

    print("--- Assignment ---")
    print(f"model_id:        {assignment.model_id}")
    print(f"layers:          {assignment.layers}")
    print(f"n_cpu_moe:       {assignment.n_cpu_moe}")
    print(f"tensor_split:    {assignment.tensor_split}")
    print(f"predicted_tok_s: {assignment.predicted_tok_s:.2f}")
    print(f"rationale:       {assignment.rationale}\n")

    print("--- Per-node footprint check ---")
    for node_id in assignment.layers:
        footprint = node_footprint_mb(assignment, MOCK_MODEL_SPEC, node_id)
        print(f"{node_id}: {footprint:.0f} MB used")

    print(
        f"\nfits() overall: {fits(assignment, MOCK_PROFILES, MOCK_METRICS, MOCK_MODEL_SPEC)}"
    )

    # A second call demonstrates SCH-4's re-plan path: same function, fresh
    # inputs. Here we simulate office-02 dying mid-demo.
    print("\n--- Re-plan after office-02 drops ---")
    remaining_profiles = [p for p in MOCK_PROFILES if p.id != "office-02"]
    remaining_metrics = [m for m in MOCK_METRICS if m.node_id != "office-02"]
    replanned = plan(remaining_profiles, remaining_metrics, MOCK_MODEL_SPEC)
    print(f"layers:          {replanned.layers}")
    print(f"predicted_tok_s: {replanned.predicted_tok_s:.2f}")
    print(f"rationale:       {replanned.rationale}")


if __name__ == "__main__":
    run_example()
