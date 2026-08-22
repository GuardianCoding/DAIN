import httpx
import pytest

from agent.client import DainClient
from agent.tools import TOOLS, call_tool, tool_schemas

# Two nodes: one calibrated with a GPU, one uncalibrated and CPU-only. Field
# names mirror contracts.NodeProfile; the values are invented.
NODES = [
    {
        "id": "gpu-01",
        "host": "10.0.0.1",
        "cpu": "Ryzen 5 9600X",
        "cores": 12,
        "ram_total_mb": 62976,
        "ram_free_mb": 51200,
        "gpu": "RTX 5070 Ti",
        "vram_total_mb": 16384,
        "backend": "cuda",
        "mem_bandwidth_gbs": 80.0,
        "tg_tok_s": 42.1,
        "pp_tok_s": 310.0,
        "rtt_ms": 0.2,
        "state": "idle",
    },
    {
        "id": "nuc-01",
        "host": "10.0.0.5",
        "cpu": "i3-7100U",
        "cores": 4,
        "ram_total_mb": 3891,
        "ram_free_mb": 3300,
        "gpu": None,
        "vram_total_mb": 0,
        "backend": "cpu",
        "mem_bandwidth_gbs": 0.0,
        "tg_tok_s": 0.0,
        "pp_tok_s": 0.0,
        "rtt_ms": 1.1,
        "state": "idle",
    },
]

# Live telemetry for gpu-01 only, and deliberately disagreeing with the
# profile's join-time ram_free_mb so the merge is observable.
METRICS = {
    "type": "metrics",
    "nodes": [
        {
            "node_id": "gpu-01",
            "timestamp": 1.0,
            "cpu_percent": 4.0,
            "ram_free_mb": 10240,
            "gpu_percent": 12.0,
            "vram_free_mb": 15360,
            "jobs_running": 2,
        }
    ],
    "history": {},
    "llama": {},
    "llama_history": [],
    "errors": {"nuc-01": "connect timeout"},
}


def client_for(routes: dict[str, tuple[int, object]]) -> DainClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        status, body = routes[request.url.path]
        return httpx.Response(status, json=body)

    return DainClient(
        "ctl.local:8000",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_every_tool_exposes_a_well_formed_openai_schema():
    schemas = tool_schemas()

    assert len(schemas) == len(TOOLS)
    for schema in schemas:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] and function["description"]
        assert function["parameters"]["type"] == "object"
        # Absent 'required' makes small models omit arguments they should send.
        assert "required" in function["parameters"]


@pytest.mark.asyncio
async def test_cluster_status_prefers_live_telemetry_over_the_join_time_profile():
    """Both carry ram_free_mb and they mean different things: the profile's is
    frozen at join, the metric's is current. Reporting the stale one is the
    single most embarrassing thing this tool could do on stage."""
    client = client_for({"/api/nodes": (200, NODES), "/api/metrics": (200, METRICS)})
    out = await call_tool(client, "cluster_status", {})
    await client.aclose()

    assert "ram_free=10.0GiB" in out  # the metric, 10240 MiB
    assert "50.0GiB" not in out  # not the profile's 51200 MiB
    assert "jobs_running=2" in out


@pytest.mark.asyncio
async def test_cluster_status_calls_an_unmeasured_node_uncalibrated_not_zero():
    """tg_tok_s is 0.0 until SCH-1 runs. Rendered literally, a model will
    happily tell the audience the NUC generates zero tokens per second."""
    client = client_for({"/api/nodes": (200, NODES), "/api/metrics": (200, METRICS)})
    out = await call_tool(client, "cluster_status", {})
    await client.aclose()

    assert "decode=uncalibrated" in out
    assert "decode=0.0tok/s" not in out


@pytest.mark.asyncio
async def test_cluster_status_flags_a_node_with_no_live_telemetry():
    client = client_for({"/api/nodes": (200, NODES), "/api/metrics": (200, METRICS)})
    out = await call_tool(client, "cluster_status", {})
    await client.aclose()

    nuc_line = next(line for line in out.splitlines() if line.startswith("nuc-01"))
    assert "no live telemetry" in nuc_line
    assert "connect timeout" in out


@pytest.mark.asyncio
async def test_cluster_status_reports_an_empty_pool_as_a_fact_not_an_error():
    client = client_for(
        {"/api/nodes": (200, []), "/api/metrics": (200, {"nodes": [], "errors": {}})}
    )
    out = await call_tool(client, "cluster_status", {})
    await client.aclose()

    assert "no nodes" in out.lower()


@pytest.mark.asyncio
async def test_a_tool_that_503s_returns_the_reason_instead_of_raising():
    """A 503 is normal, not exceptional: nodes calibrate, models load. The loop
    must survive it, so the reason has to come back as an ordinary result."""
    detail = "no node has a measured tg_tok_s — profile the cluster before planning"
    client = client_for({"/api/plan": (503, {"detail": detail})})
    out = await call_tool(client, "plan_placement", {"model": "castoff"})
    await client.aclose()

    assert "profile the cluster before planning" in out


@pytest.mark.asyncio
async def test_an_unknown_model_comes_back_with_the_keys_that_would_have_worked():
    client = client_for({"/api/plan": (404, {"detail": "unknown model 'gpt-oss-20b'"})})
    out = await call_tool(client, "plan_placement", {"model": "gpt-oss-20b"})
    await client.aclose()

    assert "gpt-oss-20b" in out
    assert "castoff" in out  # the tool appends the valid keys


@pytest.mark.asyncio
async def test_plan_placement_renders_the_split_and_the_rationale():
    plan = {
        "model_id": "castoff",
        "layers": {"office-01": [0, 11], "nuc-01": [12, 23]},
        "n_cpu_moe": {},
        "tensor_split": [0.6, 0.4],
        "predicted_tok_s": 9.4,
        "rationale": "office-01 is twice the bandwidth of nuc-01",
    }
    client = client_for({"/api/plan": (200, plan)})
    out = await call_tool(client, "plan_placement", {"model": "castoff"})
    await client.aclose()

    assert "office-01" in out and "layers 0-11" in out
    assert "9.4" in out
    assert "twice the bandwidth" in out


@pytest.mark.asyncio
async def test_an_unknown_tool_name_names_the_tools_that_exist():
    client = client_for({})
    out = await call_tool(client, "reboot_everything", {})
    await client.aclose()

    assert "reboot_everything" in out
    assert "cluster_status" in out


@pytest.mark.asyncio
async def test_a_missing_required_argument_is_reported_not_raised():
    client = client_for({})
    out = await call_tool(client, "plan_placement", {})
    await client.aclose()

    assert "model" in out
