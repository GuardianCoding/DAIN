import json

import httpx
import pytest

from agent.client import DainClient
from agent.tools import TOOLS, call_tool, tool_schemas
from node.sandbox import DEFAULT_ALLOWLIST

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


def job_ctl(job: dict) -> tuple[DainClient, list[dict]]:
    """A ctl that accepts one job and returns `job` when polled."""
    submitted: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/nodes":
            return httpx.Response(200, json=NODES)
        if path == "/api/jobs" and request.method == "POST":
            submitted.append(json.loads(request.content))
            return httpx.Response(201, json={"id": "job-1", "status": "queued"})
        return httpx.Response(200, json={"id": "job-1", **job})

    client = DainClient(
        "ctl.local:8000",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        poll_interval_s=0.001,
    )
    return client, submitted


@pytest.mark.asyncio
async def test_search_fans_out_across_every_available_node():
    """Nobody centralised the corpus; each machine searches its own disk. A
    fan-out of 1 would quietly search one node and look identical."""
    client, submitted = job_ctl(
        {
            "status": "done",
            "result": {
                "shards": [],
                "errors": [],
                "hits": [
                    {"source": "gpu-01:/srv/notes.md", "score": 0.91, "snippet": "hi"}
                ],
                "nodes_searched": ["gpu-01", "nuc-01"],
            },
        }
    )
    out = await call_tool(client, "search_files", {"query": "rpc handshake"})
    await client.aclose()

    assert submitted[0]["fanout"] == len(NODES)
    assert submitted[0]["payload"]["query"] == "rpc handshake"
    assert "gpu-01:/srv/notes.md" in out
    assert "0.91" in out


@pytest.mark.asyncio
async def test_search_reports_the_hits_it_got_even_when_a_node_failed():
    """Partial failure is the common case: one node's index is cold. Returning
    only the error throws away results the cluster actually produced."""
    client, _ = job_ctl(
        {
            "status": "failed",
            "result": {
                "shards": [],
                "errors": [{"shard_index": 1, "error": "HTTP 409: index is not ready"}],
                "hits": [{"source": "gpu-01:/srv/a.md", "score": 0.8, "snippet": "x"}],
                "nodes_searched": ["gpu-01"],
            },
        }
    )
    out = await call_tool(client, "search_files", {"query": "anything"})
    await client.aclose()

    assert "gpu-01:/srv/a.md" in out
    assert "index is not ready" in out


@pytest.mark.asyncio
async def test_search_without_a_query_says_so():
    client, submitted = job_ctl({"status": "done", "result": {}})
    out = await call_tool(client, "search_files", {})
    await client.aclose()

    assert "query" in out
    assert submitted == []


@pytest.mark.asyncio
async def test_run_command_returns_stdout_and_the_node_that_ran_it():
    client, submitted = job_ctl(
        {
            "status": "done",
            "result": {
                "shards": [
                    {
                        "shard_index": 0,
                        "node_id": "gpu-01",
                        "result": {
                            "ok": True,
                            "exit_code": 0,
                            "stdout": "total 4\nnotes.md\n",
                            "stderr": "",
                        },
                    }
                ],
                "errors": [],
            },
        }
    )
    out = await call_tool(
        client, "run_command", {"argv": ["ls", "-la"], "node": "gpu-01"}
    )
    await client.aclose()

    assert submitted[0]["node_id"] == "gpu-01"
    assert submitted[0]["payload"]["argv"] == ["ls", "-la"]
    assert "notes.md" in out
    assert "exit 0" in out


@pytest.mark.parametrize("program", ["python", "curl", "rm", "bash"])
@pytest.mark.asyncio
async def test_a_program_outside_the_sandbox_allowlist_never_reaches_a_node(
    program: str,
):
    """The sandbox is the real boundary and rejects these itself. Checking here
    too turns a wasted job dispatch into an answer the model can act on."""
    client, submitted = job_ctl({"status": "done", "result": {}})
    out = await call_tool(client, "run_command", {"argv": [program, "x"]})
    await client.aclose()

    assert submitted == []
    assert program in out
    assert "grep" in out  # the reply lists what is allowed
    assert set(DEFAULT_ALLOWLIST) >= {"grep", "ls", "cat"}


@pytest.mark.asyncio
async def test_run_command_with_no_argv_says_so():
    client, submitted = job_ctl({"status": "done", "result": {}})
    out = await call_tool(client, "run_command", {})
    await client.aclose()

    assert "argv" in out
    assert submitted == []


@pytest.mark.asyncio
async def test_ask_pool_labels_each_answer_with_the_machine_that_gave_it():
    client, submitted = job_ctl(
        {
            "status": "done",
            "result": {
                "shards": [
                    {
                        "shard_index": 0,
                        "node_id": "gpu-01",
                        "result": {"text": "an answer", "tok_s": 30.2},
                    }
                ],
                "errors": [],
            },
        }
    )
    out = await call_tool(
        client, "ask_pool", {"prompts": ["summarise A", "summarise B"]}
    )
    await client.aclose()

    assert len(submitted) == 2
    assert "gpu-01" in out
    assert "an answer" in out
    assert "30.2" in out


@pytest.mark.asyncio
async def test_ask_pool_needs_a_list_of_prompts():
    client, submitted = job_ctl({"status": "done", "result": {}})
    out = await call_tool(client, "ask_pool", {"prompts": "just a string"})
    await client.aclose()

    assert "list" in out
    assert submitted == []


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
