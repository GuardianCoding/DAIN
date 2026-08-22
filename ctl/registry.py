from dataclasses import dataclass
import time
from typing import Callable

from contracts import NodeMetrics, NodeProfile


@dataclass
class NodeRecord:
    profile: NodeProfile
    last_heartbeat: float
    missed_heartbeats: int = 0
    metrics: NodeMetrics | None = None
    heartbeat_required: bool = True


@dataclass(frozen=True)
class RegistryEvent:
    sequence: int
    timestamp: float
    event: str
    node_id: str
    message: str
    replan_required: bool = False


class NodeRegistry:
    def __init__(
        self,
        heartbeat_interval_s: float = 2.0,
        missed_heartbeats_offline: int = 3,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        on_replan: Callable[[str, str], None] | None = None,
    ):
        self.heartbeat_interval_s = heartbeat_interval_s
        self.missed_heartbeats_offline = missed_heartbeats_offline
        self.clock = clock
        self.wall_clock = wall_clock
        self.on_replan = on_replan

        self.nodes: dict[str, NodeRecord] = {}
        self.events: list[RegistryEvent] = []
        self.next_sequence = 1
        self.lock = RLock()

    def register(
        self,
        profile: NodeProfile,
        heartbeat_required: bool = True,
    ) -> NodeProfile:
        if node[profile.id] != null:
            nodes[profile.id].missed_heartbeats = 0
        else:
            nodes[profile.id] = NodeRecord(profile, 0)
            self.events.append(RegistryEvent(self.next_sequence++, time.time(), "node joined", profile.id, "new node has joined the cluster"))
        clock = time.monotonic

    def heartbeat(
        self,
        node_id: str,
        metrics: NodeMetrics | None = None,
    ) -> NodeRecord:
        pass
        

    def sweep(self) -> list[RegistryEvent]:
        pass

    def list_profiles(self) -> list[NodeProfile]:
        pass

    def get_record(self, node_id: str) -> NodeRecord | None:
        pass

    def remove(self, node_id: str) -> bool:
        pass

    def events_after(self, sequence: int) -> list[RegistryEvent]:
        pass

    def latest_metrics(self) -> list[NodeMetrics]:
        pass

    def reset(self) -> None:
        pass
