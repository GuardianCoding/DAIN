from sched.cost import COMPUTE_OVERHEAD_MB, fits
from sched.plan import plan

from contracts import NodeMetrics, NodeProfile

# Every memory figure below is written as "usable + COMPUTE_OVERHEAD_MB",
# because usable_mem_mb() subtracts llama.cpp's per-node compute-buffer
# allowance before any layer is placed. Stating the usable number and adding
# the overhead back keeps the arithmetic in each test readable, and means
# these fixtures stay correct if the constant is ever retuned.


def make_profile(
    node_id: str,
    *,
    speed: float,
    state: str = "idle",
) -> NodeProfile:
    return NodeProfile(
        id=node_id,
        host=f"{node_id}.local",
        cpu="test cpu",
        cores=4,
        ram_total_mb=1_000 + COMPUTE_OVERHEAD_MB,
        ram_free_mb=1_000 + COMPUTE_OVERHEAD_MB,
        gpu=None,
        vram_total_mb=0,
        backend="cpu",
        mem_bandwidth_gbs=10.0,
        tg_tok_s=speed,
        pp_tok_s=speed,
        rtt_ms=0.1,
        state=state,
    )


def make_metrics(node_id: str, *, free_mb: int) -> NodeMetrics:
    return NodeMetrics(
        node_id=node_id,
        timestamp=0.0,
        cpu_percent=0.0,
        ram_free_mb=free_mb,
        gpu_percent=None,
        vram_free_mb=None,
    )


def layer_count(layers: tuple[int, int]) -> int:
    first, last = layers
    return max(0, last - first + 1)


def test_plan_excludes_offline_nodes() -> None:
    profiles = [
        make_profile("idle-01", speed=1.0),
        make_profile("offline-01", speed=100.0, state="offline"),
    ]
    metrics = [
        make_metrics("idle-01", free_mb=1_000 + COMPUTE_OVERHEAD_MB),
        make_metrics("offline-01", free_mb=1_000 + COMPUTE_OVERHEAD_MB),
    ]
    model = {
        "model_id": "test-model",
        "total_layers": 4,
        "file_size_mb": 400,
        "kv_mb_per_layer": 0.0,
    }

    assignment = plan(profiles, metrics, model)

    assert assignment.layers == {"idle-01": (0, 3)}


def test_repair_preserves_layers_and_updates_tensor_split() -> None:
    profiles = [
        make_profile("a", speed=9.0),
        make_profile("b", speed=1.0),
    ]
    metrics = [
        # 300 usable holds 3 of the 10 100-MiB layers; 1000 holds the other 7.
        make_metrics("a", free_mb=300 + COMPUTE_OVERHEAD_MB),
        make_metrics("b", free_mb=1_000 + COMPUTE_OVERHEAD_MB),
    ]
    model = {
        "model_id": "test-model",
        "total_layers": 10,
        "file_size_mb": 1_000,
        "kv_mb_per_layer": 0.0,
    }

    assignment = plan(profiles, metrics, model)

    assert sum(layer_count(value) for value in assignment.layers.values()) == 10
    assert layer_count(assignment.layers["a"]) == 3
    assert layer_count(assignment.layers["b"]) == 7
    assert assignment.tensor_split == [0.3, 0.7]
    assert fits(assignment, profiles, metrics, model)


def test_repair_counts_kv_memory_when_selecting_donor() -> None:
    profiles = [
        make_profile("a", speed=8.0),
        make_profile("b", speed=2.0),
        make_profile("c", speed=1.0),
    ]
    metrics = [
        # Each layer costs 100 MiB of weights + 100 MiB of KV, so 250 usable
        # buys exactly one layer on b — the point the donor choice turns on.
        make_metrics("a", free_mb=400 + COMPUTE_OVERHEAD_MB),
        make_metrics("b", free_mb=250 + COMPUTE_OVERHEAD_MB),
        make_metrics("c", free_mb=1_000 + COMPUTE_OVERHEAD_MB),
    ]
    model = {
        "model_id": "test-model",
        "total_layers": 6,
        "file_size_mb": 600,
        "kv_mb_per_layer": 100.0,
    }

    assignment = plan(profiles, metrics, model)

    assert layer_count(assignment.layers["b"]) == 1
    assert layer_count(assignment.layers["c"]) == 3
    assert fits(assignment, profiles, metrics, model)
