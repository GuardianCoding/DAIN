from sched.cost import fits
from sched.plan import plan

from contracts import NodeMetrics, NodeProfile


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
        ram_total_mb=2_000,
        ram_free_mb=2_000,
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
        make_metrics("idle-01", free_mb=1_000),
        make_metrics("offline-01", free_mb=1_000),
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
        make_metrics("a", free_mb=300),
        make_metrics("b", free_mb=1_000),
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
        make_metrics("a", free_mb=400),
        make_metrics("b", free_mb=250),
        make_metrics("c", free_mb=1_000),
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
