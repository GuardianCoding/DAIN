# contracts.py — the whole API surface. Everyone imports from here.
from dataclasses import dataclass
from typing import Literal, Optional

NodeState = Literal["joining", "idle", "computing", "degraded", "offline"]

@dataclass
class NodeProfile:            # produced by the node agent, consumed by the scheduler
    id: str; host: str
    cpu: str; cores: int
    ram_total_mb: int; ram_free_mb: int
    gpu: Optional[str]; vram_total_mb: int
    backend: str                    # "vulkan" | "cpu" | "cuda"
    mem_bandwidth_gbs: float        # measured
    tg_tok_s: float                 # measured decode on the calibration model
    pp_tok_s: float                 # measured prefill on the calibration model
    rtt_ms: float
    state: NodeState = "joining"

@dataclass
class Assignment:             # the only thing the scheduler produces
    model_id: str
    layers: dict[str, tuple[int, int]]     # node_id -> (first, last)
    n_cpu_moe: dict[str, int]              # node_id -> expert layers pushed to RAM
    tensor_split: list[float]              # ordered to match the --rpc list
    predicted_tok_s: float
    rationale: str                         # plain English; goes on the dashboard

@dataclass
class Job:
    id: str
    kind: Literal["infer","exec","index","search","bench"]
    payload: dict
    node_id: Optional[str] = None          # None = scheduler picks
    status: Literal["queued","running","done","failed"] = "queued"
    result: Optional[dict] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None