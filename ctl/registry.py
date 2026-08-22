import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

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
        with self.lock:
            existing = self.nodes.get(profile.id)
            if existing is not None:
                profile.state = existing.profile.state
                existing.profile = profile
                existing.last_heartbeat = self.clock()
                existing.missed_heartbeats = 0
                existing.heartbeat_required = heartbeat_required
                return existing.profile

            profile.state = "joining"

            self.nodes[profile.id] = NodeRecord(
                profile, self.clock(), heartbeat_required=heartbeat_required
            )

            self.events.append(
                RegistryEvent(
                    sequence=self.next_sequence,
                    timestamp=self.wall_clock(),
                    event="joined",
                    node_id=profile.id,
                    message=f"Node {profile.id} joined the cluster",
                )
            )

            self.next_sequence += 1
            return profile

    def heartbeat(
        self,
        node_id: str,
        metrics: NodeMetrics | None = None,
    ) -> NodeRecord:
        with self.lock:
            record = self.nodes.get(node_id)

            if record is None:
                raise KeyError(node_id)

            if metrics is not None and metrics.node_id != node_id:
                raise ValueError(f"Metrics belong to {metrics.node_id}, not {node_id}")

            was_offline = record.profile.state == "offline"

            record.last_heartbeat = self.clock()
            record.missed_heartbeats = 0
            record.profile.state = "idle"

            if metrics is not None:
                record.metrics = metrics

            if was_offline:
                self.events.append(
                    RegistryEvent(
                        sequence=self.next_sequence,
                        timestamp=self.wall_clock(),
                        event="recovered",
                        node_id=node_id,
                        message=f"Node {node_id} recovered",
                    )
                )
                self.next_sequence += 1

        return record

    def sweep(self) -> list[RegistryEvent]:
        now = self.clock()
        emitted_events: list[RegistryEvent] = []
        replan_nodes: list[str] = []

        with self.lock:
            for node_id, record in self.nodes.items():
                if not record.heartbeat_required:
                    continue

                if record.profile.state == "offline":
                    continue

                elapsed = max(0.0, now - record.last_heartbeat)
                missed = int(elapsed // self.heartbeat_interval_s)
                record.missed_heartbeats = min(
                    missed,
                    self.missed_heartbeats_offline,
                )

                if missed < self.missed_heartbeats_offline:
                    continue

                record.profile.state = "offline"

                event = RegistryEvent(
                    sequence=self.next_sequence,
                    timestamp=self.wall_clock(),
                    event="offline",
                    node_id=node_id,
                    message=f"Node {node_id} missed its heartbeats",
                    replan_required=True,
                )

                self.events.append(event)
                emitted_events.append(event)
                self.next_sequence += 1
                replan_nodes.append(node_id)

        if self.on_replan is not None:
            for node_id in replan_nodes:
                self.on_replan(node_id, "heartbeat_timeout")

        return emitted_events

    def list_profiles(self) -> list[NodeProfile]:
        return [record.profile for record in self.nodes.values()]

    def get_record(self, node_id: str) -> NodeRecord | None:
        return self.nodes.get(node_id)

    def remove(self, node_id: str) -> bool:
        return self.nodes.pop(node_id, None) is not None

    def events_after(self, sequence: int) -> list[RegistryEvent]:
        return [event for event in self.events if event.sequence > sequence]

    def latest_metrics(self) -> list[NodeMetrics]:
        return [
            record.metrics
            for record in self.nodes.values()
            if record.metrics is not None
        ]

    def reset(self) -> None:
        self.nodes: dict[str, NodeRecord] = {}
        self.events: list[RegistryEvent] = []
        self.next_sequence = 1
        self.lock = RLock()
