import pytest

from contracts import NodeMetrics
from ctl.registry import NodeRegistry
from ctl.telemetry import TelemetryFanIn


def make_metrics(
    node_id: str = "node-01",
    *,
    timestamp: float = 0.0,
) -> NodeMetrics:
    return NodeMetrics(
        node_id=node_id,
        timestamp=timestamp,
        cpu_percent=25.0,
        ram_free_mb=6_000,
        gpu_percent=None,
        vram_free_mb=None,
        jobs_running=0,
    )


def test_ring_buffer_keeps_only_sixty_samples() -> None:
    telemetry = TelemetryFanIn(
        NodeRegistry(),
        history_limit=60,
    )

    for timestamp in range(65):
        telemetry.record(make_metrics(timestamp=float(timestamp)))

    frame = telemetry.frame()
    history = frame["history"]["node-01"]

    assert len(history) == 60
    assert history[0]["timestamp"] == 5.0
    assert history[-1]["timestamp"] == 64.0
    assert frame["nodes"][0]["timestamp"] == 64.0


def test_remove_clears_node_telemetry() -> None:
    telemetry = TelemetryFanIn(NodeRegistry())
    telemetry.record(make_metrics())

    telemetry.remove("node-01")

    assert telemetry.frame()["nodes"] == []
    assert telemetry.frame()["history"] == {}


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        TelemetryFanIn(NodeRegistry(), interval_s=0)

    with pytest.raises(ValueError):
        TelemetryFanIn(NodeRegistry(), history_limit=0)
