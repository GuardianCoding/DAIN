"""NODE-6: /infer and the LocalInference backend behind it."""

import subprocess
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from contracts import NodeProfile
from node import dain_node
from node.auth import sign_job_request
from node.infer import (
    DEFAULT_LLAMA_PORT,
    InferenceUnavailableError,
    LocalInference,
    _normalise,
)

POOL_SECRET = "test-pool-secret"
NODE_ID = "fedora-test"
CTL = "127.0.0.1:8000"


def make_profile() -> NodeProfile:
    return NodeProfile(
        id=NODE_ID,
        host="192.168.50.20",
        cpu="test cpu",
        cores=4,
        ram_total_mb=8192,
        ram_free_mb=6144,
        gpu=None,
        vram_total_mb=0,
        backend="cpu",
        mem_bandwidth_gbs=0.0,
        tg_tok_s=0.0,
        pp_tok_s=0.0,
        rtt_ms=0.4,
    )


def transport(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def healthy(body: dict):
    """A llama-server that is up and answers one completion with `body`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json=body)

    return handler


# --------------------------------------------------------------------------
# Backend resolution
# --------------------------------------------------------------------------


class TestBackendResolution:
    def test_unconfigured_node_serves_no_inference(self):
        assert LocalInference().configured is False
        assert LocalInference().supervises is False

    def test_model_file_means_this_node_supervises_a_server(self):
        engine = LocalInference(model_file="/models/replica.gguf")

        assert engine.supervises is True
        assert engine.base_url == f"http://127.0.0.1:{DEFAULT_LLAMA_PORT}"

    def test_explicit_endpoint_wins_over_a_local_model(self):
        # "Use this exact server" is the more specific instruction, so a node
        # told to forward must not also start a competing local server.
        engine = LocalInference(
            endpoint="http://gpu-01:8080",
            model_file="/models/replica.gguf",
        )

        assert engine.supervises is False
        assert engine.base_url == "http://gpu-01:8080"

    def test_trailing_slash_does_not_double_up_in_urls(self):
        assert (
            LocalInference(endpoint="http://gpu-01:8080/").base_url
            == "http://gpu-01:8080"
        )

    def test_from_environment_reads_the_documented_names(self, monkeypatch):
        monkeypatch.setenv("DAIN_LLAMA_ENDPOINT", "http://gpu-01:8080")
        monkeypatch.setenv("DAIN_LLAMA_PORT", "9090")
        monkeypatch.setenv("DAIN_INFER_CONTEXT", "4096")

        engine = LocalInference.from_environment()

        assert engine.endpoint == "http://gpu-01:8080"
        assert engine.port == 9090
        assert engine.context == 4096

    def test_blank_env_var_is_treated_as_unset(self, monkeypatch):
        # `export DAIN_LLAMA_ENDPOINT=` is a plausible way to "turn it off".
        monkeypatch.setenv("DAIN_LLAMA_ENDPOINT", "   ")

        assert LocalInference.from_environment().endpoint is None

    def test_non_numeric_port_falls_back_instead_of_crashing(self, monkeypatch):
        monkeypatch.setenv("DAIN_LLAMA_PORT", "not-a-port")

        assert LocalInference.from_environment().port == DEFAULT_LLAMA_PORT


class TestServerCommand:
    def test_refuses_to_build_a_command_for_a_missing_model(self, tmp_path):
        engine = LocalInference(model_file=str(tmp_path / "absent.gguf"))

        with pytest.raises(InferenceUnavailableError, match="model file not found"):
            engine.server_command()

    def test_binds_loopback_only(self, tmp_path):
        # llama-server has no authentication; a wildcard bind hands this
        # machine's compute to anything that can reach it.
        model = tmp_path / "replica.gguf"
        model.write_bytes(b"gguf")
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)

        command = LocalInference(
            model_file=str(model), bin_dir=str(tmp_path)
        ).server_command()

        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert "0.0.0.0" not in command

    def test_start_is_a_no_op_without_a_model(self, monkeypatch):
        started: list[list[str]] = []
        monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: started.append(argv))

        LocalInference().start()

        assert started == []


# --------------------------------------------------------------------------
# Completion
# --------------------------------------------------------------------------


class TestComplete:
    @pytest.mark.asyncio
    async def test_unconfigured_node_explains_both_ways_to_fix_it(self):
        with pytest.raises(InferenceUnavailableError) as excinfo:
            await LocalInference().complete({"prompt": "hi"})

        message = str(excinfo.value)
        assert "DAIN_INFER_MODEL" in message
        assert "DAIN_LLAMA_ENDPOINT" in message

    @pytest.mark.asyncio
    async def test_empty_prompt_is_rejected_before_any_request(self):
        sent: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(str(request.url))
            return httpx.Response(200, json={})

        engine = LocalInference(endpoint="http://gpu-01:8080", client=transport(handler))

        with pytest.raises(ValueError, match="prompt must not be empty"):
            await engine.complete({"prompt": "   "})
        assert sent == []

    @pytest.mark.asyncio
    async def test_non_integer_max_tokens_is_rejected(self):
        engine = LocalInference(
            endpoint="http://gpu-01:8080",
            client=transport(healthy({})),
        )

        with pytest.raises(ValueError, match="max_tokens must be an integer"):
            await engine.complete({"prompt": "hi", "max_tokens": "lots"})

    @pytest.mark.asyncio
    async def test_returns_flattened_text_and_speed(self):
        engine = LocalInference(
            endpoint="http://gpu-01:8080",
            client=transport(
                healthy(
                    {
                        "model": "qwen3-4b",
                        "choices": [
                            {
                                "message": {"content": "Idle office PCs hum."},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                        "timings": {"predicted_per_second": 24.5},
                    }
                )
            ),
        )

        result = await engine.complete({"prompt": "haiku please"})

        assert result["text"] == "Idle office PCs hum."
        assert result["model"] == "qwen3-4b"
        assert result["completion_tokens"] == 7
        assert result["tok_s"] == 24.5
        assert result["endpoint"] == "http://gpu-01:8080"

    @pytest.mark.asyncio
    async def test_a_dead_server_is_unavailable_not_a_crash(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        engine = LocalInference(
            endpoint="http://gpu-01:8080",
            client=transport(handler),
            ready_timeout_s=0.01,
        )

        with pytest.raises(InferenceUnavailableError):
            await engine.complete({"prompt": "hi"})


class TestNormalise:
    def test_missing_timings_still_yields_a_speed(self):
        # llama.cpp has shipped builds without a timings block; a successful
        # generation must not become a failed job over a missing field.
        result = _normalise(
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"completion_tokens": 10},
            },
            elapsed_s=2.0,
            endpoint="http://x",
        )

        assert result["tok_s"] == 5.0

    def test_empty_envelope_does_not_raise(self):
        result = _normalise({}, elapsed_s=1.0, endpoint="http://x")

        assert result["text"] == ""
        assert result["tok_s"] is None


# --------------------------------------------------------------------------
# The HTTP route
# --------------------------------------------------------------------------


def signed_body(payload: dict, *, kind: str = "infer") -> dict:
    # A real clock, not 0: verify_job_request enforces a replay window, so a
    # fixed issued_at makes every request 403 on staleness and the route under
    # test never runs.
    issued_at = int(time.time())
    body = {
        "job_id": "job-1",
        "kind": kind,
        "payload": payload,
        "shard_index": 0,
        "shard_count": 1,
        "issued_at": issued_at,
    }
    body["signature"] = sign_job_request(
        POOL_SECRET,
        job_id=body["job_id"],
        kind=kind,
        payload=payload,
        shard_index=0,
        shard_count=1,
        issued_at=issued_at,
    )
    return body


def configure_with(engine: LocalInference) -> TestClient:
    dain_node.configure(
        make_profile(),
        ctl=CTL,
        pool_secret=POOL_SECRET,
        inference=engine,
    )
    return TestClient(dain_node.app)


@pytest.fixture(autouse=True)
def clear_agent():
    if getattr(dain_node.app.state, "agent", None) is not None:
        del dain_node.app.state.agent
    yield
    if getattr(dain_node.app.state, "agent", None) is not None:
        del dain_node.app.state.agent


class TestInferRoute:
    def test_infer_kind_passes_request_validation(self):
        """The regression this route existed to expose.

        LocalJobRequest.kind was Literal["exec", "index", "search"], so a
        dispatched infer job was rejected by request validation before routing
        — which is why adding the route alone would not have been enough.
        """
        kind = dain_node.LocalJobRequest.model_fields["kind"]
        assert "infer" in kind.annotation.__args__

    def test_unconfigured_node_returns_503_not_404(self):
        client = configure_with(LocalInference())

        response = client.post("/infer", json=signed_body({"prompt": "hi"}))

        assert response.status_code == 503
        assert "DAIN_INFER_MODEL" in response.json()["detail"]

    def test_wrong_kind_for_the_route_is_422(self):
        client = configure_with(LocalInference())

        response = client.post("/infer", json=signed_body({}, kind="exec"))

        assert response.status_code == 422

    def test_unsigned_request_is_rejected(self):
        client = configure_with(LocalInference())
        body = signed_body({"prompt": "hi"})
        body["signature"] = "0" * 64

        response = client.post("/infer", json=body)

        assert response.status_code == 403

    def test_empty_prompt_is_422_not_503(self):
        client = configure_with(
            LocalInference(endpoint="http://gpu-01:8080", client=transport(healthy({})))
        )

        response = client.post("/infer", json=signed_body({"prompt": ""}))

        assert response.status_code == 422

    def test_successful_generation_carries_node_and_shard_identity(self):
        client = configure_with(
            LocalInference(
                endpoint="http://gpu-01:8080",
                client=transport(
                    healthy(
                        {
                            "choices": [{"message": {"content": "42"}}],
                            "usage": {"completion_tokens": 1},
                        }
                    )
                ),
            )
        )

        response = client.post("/infer", json=signed_body({"prompt": "hi"}))

        assert response.status_code == 200
        result = response.json()["result"]
        assert result["node_id"] == NODE_ID
        assert result["shard_index"] == 0
        assert result["shard_count"] == 1
        assert result["text"] == "42"
