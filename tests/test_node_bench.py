"""INF-6 / SCH-1: /bench and the llama-bench parse behind it."""

import json
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from node import bench as bench_module
from node import dain_node
from node.auth import sign_job_request
from node.bench import (
    BENCH_TIMEOUT_S,
    BenchUnavailableError,
    LocalBench,
    measure,
    parse_bench_json,
)

from contracts import NodeProfile

POOL_SECRET = "test-pool-secret"
NODE_ID = "office-01"
CTL = "127.0.0.1:8000"

# What `llama-bench -p 512 -n 128 -r 3 -o json` prints: one row per phase.
PREFILL_ROW = {
    "build_commit": "abc1234",
    "model_filename": "qwen3-0.6b.gguf",
    "model_type": "qwen3 0.6B Q4_K_M",
    "n_gpu_layers": 0,
    "n_threads": 4,
    "n_prompt": 512,
    "n_gen": 0,
    "avg_ts": 310.5,
    "stddev_ts": 4.2,
    "reps": 3,
}
DECODE_ROW = {
    **PREFILL_ROW,
    "n_prompt": 0,
    "n_gen": 128,
    "avg_ts": 42.1,
    "stddev_ts": 0.8,
}
BENCH_JSON = json.dumps([PREFILL_ROW, DECODE_ROW])


def make_profile() -> NodeProfile:
    return NodeProfile(
        id=NODE_ID,
        host="192.168.50.11",
        cpu="Intel Core i7-6700",
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


def usable_bench(tmp_path) -> LocalBench:
    """A LocalBench whose binary and model both exist on disk."""
    model = tmp_path / "calibration.gguf"
    model.write_bytes(b"gguf")
    binary = tmp_path / "llama-bench"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    return LocalBench(model_file=str(model), bin_dir=str(tmp_path))


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


class TestParseBenchJson:
    def test_prefill_and_decode_land_in_the_right_fields(self):
        result = parse_bench_json(BENCH_JSON)

        assert result.pp_tok_s == 310.5
        assert result.tg_tok_s == 42.1

    def test_row_order_does_not_decide_which_phase_is_which(self):
        """The bug this guards against is a silent 10x error.

        llama-bench has changed row order between releases. Keying on position
        rather than n_prompt/n_gen would report a 310 tok/s prefill as the
        decode speed, and the cost model would believe it.
        """
        result = parse_bench_json(json.dumps([DECODE_ROW, PREFILL_ROW]))

        assert result.pp_tok_s == 310.5
        assert result.tg_tok_s == 42.1

    def test_carries_the_build_commit(self):
        # llama.cpp's RPC protocol has no version negotiation, so which commit
        # produced a number is part of the number.
        assert parse_bench_json(BENCH_JSON).build_commit == "abc1234"

    def test_stddev_is_kept(self):
        result = parse_bench_json(BENCH_JSON)

        assert result.pp_stddev == 4.2
        assert result.tg_stddev == 0.8

    def test_decode_only_run_still_parses(self):
        result = parse_bench_json(json.dumps([DECODE_ROW]))

        assert result.tg_tok_s == 42.1
        assert result.pp_tok_s is None

    def test_non_json_output_is_reported_as_such(self):
        with pytest.raises(BenchUnavailableError, match="did not emit JSON"):
            parse_bench_json("ggml_init: using CPU backend\n")

    def test_empty_array_is_not_a_measurement(self):
        with pytest.raises(BenchUnavailableError, match="no measurements"):
            parse_bench_json("[]")

    def test_rows_with_neither_phase_are_rejected(self):
        with pytest.raises(BenchUnavailableError, match="no prefill or decode"):
            parse_bench_json(json.dumps([{"n_prompt": 0, "n_gen": 0}]))


# --------------------------------------------------------------------------
# Configuration and command
# --------------------------------------------------------------------------


class TestConfiguration:
    def test_unconfigured_node_cannot_bench(self):
        assert LocalBench().configured is False
        assert LocalBench().available is False

    def test_bench_model_env_wins_over_infer_model(self, monkeypatch):
        monkeypatch.setenv("DAIN_BENCH_MODEL", "/models/calibration.gguf")
        monkeypatch.setenv("DAIN_INFER_MODEL", "/models/replica.gguf")

        assert LocalBench.from_environment().model_file == "/models/calibration.gguf"

    def test_falls_back_to_the_served_model(self, monkeypatch):
        # A fan-out node should need one variable, not two, and benching what
        # it actually serves is the useful default.
        monkeypatch.delenv("DAIN_BENCH_MODEL", raising=False)
        monkeypatch.setenv("DAIN_INFER_MODEL", "/models/replica.gguf")

        assert LocalBench.from_environment().model_file == "/models/replica.gguf"

    def test_command_matches_the_calibration_probe(self, tmp_path):
        engine = usable_bench(tmp_path)

        command = engine.command(
            engine.model_file, repetitions=3, prompt_tokens=512, gen_tokens=128
        )

        assert command[command.index("-p") + 1] == "512"
        assert command[command.index("-n") + 1] == "128"
        assert command[command.index("-r") + 1] == "3"
        # Without -o json the parse has nothing to read.
        assert command[command.index("-o") + 1] == "json"

    def test_missing_model_file_is_unavailable(self, tmp_path):
        binary = tmp_path / "llama-bench"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        engine = LocalBench(
            model_file=str(tmp_path / "absent.gguf"), bin_dir=str(tmp_path)
        )

        with pytest.raises(BenchUnavailableError, match="model file not found"):
            engine.command(
                engine.model_file, repetitions=1, prompt_tokens=8, gen_tokens=8
            )

    def test_missing_binary_is_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bench_module.shutil, "which", lambda name: None)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"gguf")
        engine = LocalBench(model_file=str(model), bin_dir=str(tmp_path / "absent"))

        with pytest.raises(BenchUnavailableError, match="no llama-bench"):
            engine.command(str(model), repetitions=1, prompt_tokens=8, gen_tokens=8)


class TestPayloadValidation:
    @pytest.mark.asyncio
    async def test_unconfigured_node_names_both_env_vars(self):
        with pytest.raises(BenchUnavailableError) as excinfo:
            await LocalBench().run({})

        message = str(excinfo.value)
        assert "DAIN_BENCH_MODEL" in message
        assert "DAIN_INFER_MODEL" in message

    @pytest.mark.asyncio
    async def test_non_integer_repetitions_is_a_caller_error(self, tmp_path):
        with pytest.raises(ValueError, match="repetitions must be an integer"):
            await usable_bench(tmp_path).run({"repetitions": "three"})

    @pytest.mark.asyncio
    async def test_repetitions_are_capped(self, tmp_path):
        with pytest.raises(ValueError, match="between 1 and"):
            await usable_bench(tmp_path).run({"repetitions": 9999})

    @pytest.mark.asyncio
    async def test_zero_gen_tokens_is_rejected(self, tmp_path):
        # -n 0 produces no decode row, so tg_tok_s would come back None and
        # SCH-1 would be no better off than before.
        with pytest.raises(ValueError, match="between 1 and"):
            await usable_bench(tmp_path).run({"gen_tokens": 0})


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    async def test_successful_run_returns_both_speeds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            bench_module.subprocess,
            "run",
            lambda *a, **kw: FakeCompleted(stdout=BENCH_JSON),
        )

        result = await usable_bench(tmp_path).run({})

        assert result["pp_tok_s"] == 310.5
        assert result["tg_tok_s"] == 42.1
        assert result["repetitions"] == 3

    @pytest.mark.asyncio
    async def test_stderr_is_surfaced_when_the_binary_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            bench_module.subprocess,
            "run",
            lambda *a, **kw: FakeCompleted(returncode=1, stderr="unsupported quant"),
        )

        with pytest.raises(BenchUnavailableError, match="unsupported quant"):
            await usable_bench(tmp_path).run({})

    @pytest.mark.asyncio
    async def test_timeout_is_unavailable_not_a_hang(self, tmp_path, monkeypatch):
        def explode(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="llama-bench", timeout=BENCH_TIMEOUT_S)

        monkeypatch.setattr(bench_module.subprocess, "run", explode)

        with pytest.raises(BenchUnavailableError, match="exceeded"):
            await usable_bench(tmp_path).run({})

    @pytest.mark.asyncio
    async def test_measure_gives_sch1_the_two_numbers_it_needs(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            bench_module.subprocess,
            "run",
            lambda *a, **kw: FakeCompleted(stdout=BENCH_JSON),
        )

        result = await measure(usable_bench(tmp_path))

        # Exactly the NodeProfile fields that are 0.0 today and that
        # predict_tok_s() raises on.
        assert result.tg_tok_s == 42.1
        assert result.pp_tok_s == 310.5


# --------------------------------------------------------------------------
# The HTTP route
# --------------------------------------------------------------------------


def signed_body(payload: dict, *, kind: str = "bench") -> dict:
    issued_at = int(time.time())
    body = {
        "job_id": "bench-1",
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


def configure_with(engine: LocalBench) -> TestClient:
    dain_node.configure(make_profile(), ctl=CTL, pool_secret=POOL_SECRET, bench=engine)
    return TestClient(dain_node.app)


@pytest.fixture(autouse=True)
def clear_agent():
    if getattr(dain_node.app.state, "agent", None) is not None:
        del dain_node.app.state.agent
    yield
    if getattr(dain_node.app.state, "agent", None) is not None:
        del dain_node.app.state.agent


class TestBenchRoute:
    def test_bench_kind_passes_request_validation(self):
        """Same regression class as /infer.

        A kind ctl can dispatch but LocalJobRequest omits is rejected by
        request validation before routing, so widening the Literal is as
        load-bearing as adding the route.
        """
        kind = dain_node.LocalJobRequest.model_fields["kind"]
        assert "bench" in kind.annotation.__args__

    def test_unconfigured_node_returns_503_not_404(self):
        client = configure_with(LocalBench())

        response = client.post("/bench", json=signed_body({}))

        assert response.status_code == 503
        assert "DAIN_BENCH_MODEL" in response.json()["detail"]

    def test_bad_payload_is_422_so_the_queue_does_not_retry_it(self, tmp_path):
        client = configure_with(usable_bench(tmp_path))

        response = client.post("/bench", json=signed_body({"repetitions": 0}))

        assert response.status_code == 422

    def test_wrong_kind_for_the_route_is_422(self):
        client = configure_with(LocalBench())

        response = client.post("/bench", json=signed_body({}, kind="exec"))

        assert response.status_code == 422

    def test_unsigned_request_is_rejected(self):
        client = configure_with(LocalBench())
        body = signed_body({})
        body["signature"] = "0" * 64

        response = client.post("/bench", json=body)

        assert response.status_code == 403

    def test_successful_bench_carries_node_and_shard_identity(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            bench_module.subprocess,
            "run",
            lambda *a, **kw: FakeCompleted(stdout=BENCH_JSON),
        )
        client = configure_with(usable_bench(tmp_path))

        response = client.post("/bench", json=signed_body({}))

        assert response.status_code == 200
        result = response.json()["result"]
        assert result["node_id"] == NODE_ID
        assert result["shard_index"] == 0
        assert result["tg_tok_s"] == 42.1
        assert result["pp_tok_s"] == 310.5

    def test_metrics_advertise_bench_availability(self, tmp_path):
        client = configure_with(usable_bench(tmp_path))

        body = client.get("/metrics").text

        assert 'node_bench_available{node_id="office-01"} 1' in body
