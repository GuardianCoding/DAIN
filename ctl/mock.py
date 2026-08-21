import os
import time
from dataclasses import asdict, replace
from typing import Any, Literal
from uuid import uuid4

from contracts import Assignment, Job, NodeMetrics, NodeProfile

MOCK_POOL_SECRET = os.getenv("DAIN_POOL_SECRET", "mock-only-secret")
JobKind = Literal["infer", "exec", "index", "search", "bench"]


gpu_01 = NodeProfile(
    id="gpu-01",
    host="192.168.50.10",
    cpu="Intel Core i7",
    cores=4,  # Replace with the actual core count after profiling.
    ram_total_mb=64 * 1024,
    ram_free_mb=46 * 1024,
    gpu="NVIDIA GeForce RTX 5070 Ti",
    vram_total_mb=16 * 1024,
    backend="cuda",
    mem_bandwidth_gbs=0.0,
    tg_tok_s=0.0,
    pp_tok_s=0.0,
    rtt_ms=0.0,
    state="idle",
)

office_01 = NodeProfile(
    id="office-01",
    host="192.168.50.11",
    cpu="Intel Core i7 vPro",
    cores=4,  # Replace with the actual core count after profiling.
    ram_total_mb=8 * 1024,
    ram_free_mb=6 * 1024,
    gpu="Intel integrated graphics",
    vram_total_mb=0,
    backend="cpu",
    mem_bandwidth_gbs=0.0,
    tg_tok_s=0.0,
    pp_tok_s=0.0,
    rtt_ms=0.4,
    state="idle",
)

office_02 = NodeProfile(
    id="office-02",
    host="192.168.50.12",
    cpu="Intel Core i7 vPro",
    cores=4,  # Replace with the actual core count after profiling.
    ram_total_mb=8 * 1024,
    ram_free_mb=6 * 1024,
    gpu="Intel integrated graphics",
    vram_total_mb=0,
    backend="cpu",
    mem_bandwidth_gbs=0.0,
    tg_tok_s=0.0,
    pp_tok_s=0.0,
    rtt_ms=0.4,
    state="idle",
)

mac_01 = NodeProfile(
    id="mac-01",
    host="192.168.50.13",
    cpu="Apple M5 Pro",
    cores=12,  # Replace with the Mac's actual core count after profiling.
    ram_total_mb=24 * 1024,
    ram_free_mb=18 * 1024,
    gpu="Apple M5 Pro integrated GPU",
    vram_total_mb=24 * 1024,  # Unified memory; never add this to RAM capacity.
    backend="metal",
    mem_bandwidth_gbs=0.0,
    tg_tok_s=0.0,
    pp_tok_s=0.0,
    rtt_ms=0.5,
    state="idle",
)

MOCK_NODES = [gpu_01, office_01, office_02, mac_01]


def make_mock_metrics(
    counter: int, nodes: list[NodeProfile] | None = None
) -> list[NodeMetrics]:
    """Create deterministic live samples without mutating node profiles."""
    timestamp = time.time()
    samples: list[NodeMetrics] = []

    for index, node in enumerate(nodes or MOCK_NODES):
        step = (counter + index) % 10
        uses_gpu = node.backend in {"cuda", "vulkan", "metal"}
        has_separate_vram = uses_gpu and node.backend != "metal"

        samples.append(
            NodeMetrics(
                node_id=node.id,
                timestamp=timestamp,
                cpu_percent=float(15 + index * 5 + step),
                ram_free_mb=max(0, int(node.ram_total_mb * 0.75) - step * 32),
                gpu_percent=float(30 + index * 5 + step) if uses_gpu else None,
                vram_free_mb=(
                    max(0, int(node.vram_total_mb * 0.75) - step * 32)
                    if has_separate_vram
                    else None
                ),
                jobs_running=1 if (counter + index) % 3 == 0 else 0,
            )
        )

    return samples


class MockControlPlane:
    """Deterministic, in-memory implementation of the frozen CP-1 API."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.nodes = {node.id: replace(node) for node in MOCK_NODES}
        self.jobs: dict[str, Job] = {}
        self.job_fanout: dict[str, int] = {}
        self.job_assignments: dict[str, list[str]] = {}
        self.metrics_counter = 0

    def list_nodes(self) -> list[NodeProfile]:
        return sorted(self.nodes.values(), key=lambda node: node.id)

    def join_node(self, profile: NodeProfile) -> NodeProfile:
        self.nodes[profile.id] = replace(profile)
        return self.nodes[profile.id]

    def remove_node(self, node_id: str) -> bool:
        return self.nodes.pop(node_id, None) is not None

    def metrics(self) -> list[NodeMetrics]:
        samples = make_mock_metrics(self.metrics_counter, self.list_nodes())
        self.metrics_counter += 1
        return samples

    def create_job(
        self,
        *,
        kind: JobKind,
        payload: dict[str, Any],
        fanout: int,
        node_id: str | None,
    ) -> Job:
        if node_id is not None and node_id not in self.nodes:
            raise KeyError(node_id)

        available = [node.id for node in self.list_nodes() if node.state != "offline"]
        assigned = [node_id] if node_id is not None else available[:fanout]

        job = Job(
            id=uuid4().hex,
            kind=kind,
            payload=payload,
            node_id=node_id,
        )
        self.jobs[job.id] = job
        self.job_fanout[job.id] = fanout
        self.job_assignments[job.id] = assigned
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def job_response(self, job: Job) -> dict[str, Any]:
        return {
            **asdict(job),
            "fanout": self.job_fanout[job.id],
            "assigned_nodes": self.job_assignments[job.id],
        }

    def plan(self, model_id: str) -> Assignment:
        nodes = self.list_nodes()
        if not nodes:
            raise RuntimeError("no nodes are available")

        total_layers = 48
        base_layers, remainder = divmod(total_layers, len(nodes))
        first_layer = 0
        layers: dict[str, tuple[int, int]] = {}

        for index, node in enumerate(nodes):
            layer_count = base_layers + (1 if index < remainder else 0)
            last_layer = first_layer + layer_count - 1
            layers[node.id] = (first_layer, last_layer)
            first_layer = last_layer + 1

        total_memory = sum(node.ram_total_mb for node in nodes)
        tensor_split = [round(node.ram_total_mb / total_memory, 4) for node in nodes]

        return Assignment(
            model_id=model_id,
            layers=layers,
            n_cpu_moe={
                node.id: 0 if node.backend in {"cuda", "metal"} else 4 for node in nodes
            },
            tensor_split=tensor_split,
            predicted_tok_s=12.5,
            rationale=(
                "Mock plan: layers are distributed evenly, with the split weighted "
                "by each node's memory capacity."
            ),
        )

    def race(self, task: str, mode: str) -> dict[str, Any]:
        node_count = max(1, len(self.nodes))
        speedups = {1: 1.0, 2: 1.9, 3: 2.6, 4: 3.3}
        speedup = 1.0 if mode == "serial" else speedups.get(node_count, 3.3)

        return {
            "race_id": uuid4().hex,
            "task": task,
            "mode": mode,
            "nodes_used": 1 if mode == "serial" else node_count,
            "speedup": speedup,
            "estimated_seconds": round(44.0 / speedup, 2),
            "status": "completed",
        }

    def event_frame(self) -> dict[str, Any]:
        return {
            "type": "event",
            "timestamp": time.time(),
            "level": "info",
            "event": "node.joined",
            "node_id": "mac-01",
            "message": "mac-01 joined the mock pool",
        }

    def flow_frame(self) -> dict[str, Any]:
        if self.jobs:
            latest_job_id = next(reversed(self.jobs))
            job = self.jobs[latest_job_id]
            target = (self.job_assignments[job.id] or ["gpu-01"])[0]
            job_id = job.id
            label = job.kind
        else:
            target = "gpu-01"
            job_id = "demo-job"
            label = "infer"

        return {
            "type": "flow",
            "timestamp": time.time(),
            "job_id": job_id,
            "source": "ctl",
            "target": target,
            "status": "running",
            "label": label,
        }


MOCK_STATE = MockControlPlane()


def reset_mock_state() -> None:
    MOCK_STATE.reset()
