import httpx
import pytest

from contracts import NodeMetrics, NodeProfile
from ctl.registry import NodeRegistry
from ctl.telemetry import TelemetryFanIn, parse_prometheus


def make_profile(
    node_id: str = "node-01",
) -> NodeProfile:
    return NodeProfile(
        id=node_id,
        host="192.168.50.11",
        cpu="test cpu",
        cores=4,
        ram_total_mb=8_192,
        ram_free_mb=6_144,
        gpu=None,
        vram_total_mb=0,
        backend="cpu",
        mem_bandwidth_gbs=20.0,
        tg_tok_s=5.0,
        pp_tok_s=10.0,
        rtt_ms=0.4,
        state="idle",
    )


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


def live_registry(
    *,
    heartbeat_required: bool = True,
) -> NodeRegistry:
    registry = NodeRegistry()

    registry.register(
        make_profile(),
        heartbeat_required=heartbeat_required,
    )
    registry.heartbeat(
        "node-01",
        make_metrics(),
    )

    return registry


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
        TelemetryFanIn(
            NodeRegistry(),
            interval_s=0,
        )

    with pytest.raises(ValueError):
        TelemetryFanIn(
            NodeRegistry(),
            timeout_s=0,
        )

    with pytest.raises(ValueError):
        TelemetryFanIn(
            NodeRegistry(),
            history_limit=0,
        )


def test_parse_prometheus_ignores_comments() -> None:
    result = parse_prometheus(
        "# HELP example Example metric\nexample 1.5\ninvalid nope\n"
    )

    assert result == {"example": 1.5}


@pytest.mark.asyncio
async def test_poll_once_collects_node_metrics() -> None:
    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert str(request.url) == "http://192.168.50.11:9100/metrics"

        return httpx.Response(
            200,
            text=(
                "dain_node_cpu_percent 42.5\n"
                "dain_node_ram_free_mb 5500\n"
                "dain_node_gpu_percent 30\n"
                "dain_node_vram_free_mb 7000\n"
                "dain_node_jobs_running 2\n"
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        telemetry = TelemetryFanIn(
            live_registry(),
            client=client,
        )

        await telemetry.poll_once()

        frame = telemetry.frame()
        node = frame["nodes"][0]

        assert node["node_id"] == "node-01"
        assert node["cpu_percent"] == 42.5
        assert node["ram_free_mb"] == 5500
        assert node["gpu_percent"] == 30.0
        assert node["vram_free_mb"] == 7000
        assert node["jobs_running"] == 2
        assert frame["errors"] == {}


@pytest.mark.asyncio
async def test_poll_failure_keeps_last_good_sample() -> None:
    async def handler(
        _request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        telemetry = TelemetryFanIn(
            live_registry(),
            client=client,
        )
        telemetry.record(make_metrics(timestamp=10.0))

        await telemetry.poll_once()

        frame = telemetry.frame()

        assert frame["nodes"][0]["timestamp"] == 10.0
        assert "node-01" in frame["errors"]


@pytest.mark.asyncio
async def test_missing_required_metric_is_reported() -> None:
    async def handler(
        _request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            text="dain_node_cpu_percent 20\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        telemetry = TelemetryFanIn(
            live_registry(),
            client=client,
        )

        await telemetry.poll_once()

        error = telemetry.frame()["errors"]["node-01"]
        assert "dain_node_ram_free_mb" in error


@pytest.mark.asyncio
async def test_non_heartbeat_mock_node_is_not_polled() -> None:
    async def handler(
        _request: httpx.Request,
    ) -> httpx.Response:
        pytest.fail("non-heartbeating mock node should not be polled")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        telemetry = TelemetryFanIn(
            live_registry(heartbeat_required=False),
            client=client,
        )

        await telemetry.poll_once()

        assert telemetry.frame()["nodes"] == []
