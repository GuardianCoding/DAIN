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

    def _emit(
        self,
        event: str,
        node_id: str,
        message: str,
        *,
        replan_required: bool = False,
    ) -> RegistryEvent:
        """Append one event. Callers already hold the lock."""
        record = RegistryEvent(
            sequence=self.next_sequence,
            timestamp=self.wall_clock(),
            event=event,
            node_id=node_id,
            message=message,
            replan_required=replan_required,
        )
        self.events.append(record)
        self.next_sequence += 1
        return record

    def register(
        self,
        profile: NodeProfile,
        heartbeat_required: bool = True,
    ) -> NodeProfile:
        with self.lock:
            existing = self.nodes.get(profile.id)
            if existing is not None:
                # Same id at a new address still moves the --rpc list, and
                # --tensor-split is positional over it, so a running head is
                # silently wrong until it restarts. Same address is NOT a
                # membership change and must stay quiet: restarting the head
                # would drop every KV cache in the cluster for nothing.
                previous_host = existing.profile.host
                moved = previous_host != profile.host

                profile.state = existing.profile.state
                existing.profile = profile
                existing.last_heartbeat = self.clock()
                existing.missed_heartbeats = 0
                existing.heartbeat_required = heartbeat_required

                if moved:
                    self._emit(
                        "readdressed",
                        profile.id,
                        f"Node {profile.id} moved from {previous_host} "
                        f"to {profile.host}",
                        replan_required=True,
                    )
                return existing.profile

            profile.state = "joining"

            self.nodes[profile.id] = NodeRecord(
                profile, self.clock(), heartbeat_required=heartbeat_required
            )

            # Expand invalidates a running head exactly as much as contract:
            # llama.cpp fixes --rpc at llama-server start, so an arriving node
            # is unusable until the head restarts.
            self._emit(
                "joined",
                profile.id,
                f"Node {profile.id} joined the cluster",
                replan_required=True,
            )
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
                # The pool grew back. The head was re-planned without this node
                # when it went offline, so it needs planning again to use it.
                self._emit(
                    "recovered",
                    node_id,
                    f"Node {node_id} recovered",
                    replan_required=True,
                )

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

                emitted_events.append(
                    self._emit(
                        "offline",
                        node_id,
                        f"Node {node_id} missed its heartbeats",
                        replan_required=True,
                    )
                )
                replan_nodes.append(node_id)

        if self.on_replan is not None:
            for node_id in replan_nodes:
                self.on_replan(node_id, "heartbeat_timeout")

        return emitted_events

    def list_profiles(self) -> list[NodeProfile]:
        with self.lock:
            return [record.profile for record in self.nodes.values()]

    def get_record(self, node_id: str) -> NodeRecord | None:
        with self.lock:
            return self.nodes.get(node_id)

    def remove(self, node_id: str) -> bool:
        with self.lock:
            if self.nodes.pop(node_id, None) is None:
                return False

            # DELETE /api/nodes/{id} used to be silent: the topology frame
            # changed but no event said why, so a consumer watching the feed
            # saw a node vanish with no reason and no replan signal.
            self._emit(
                "removed",
                node_id,
                f"Node {node_id} was removed from the cluster",
                replan_required=True,
            )
            return True

    def events_after(self, sequence: int) -> list[RegistryEvent]:
        with self.lock:
            return [event for event in self.events if event.sequence > sequence]

    def latest_metrics(self) -> list[NodeMetrics]:
        with self.lock:
            return [
                record.metrics
                for record in self.nodes.values()
                if record.metrics is not None
            ]

    def reset(self) -> None:
        with self.lock:
            self.nodes.clear()
            self.events.clear()
            self.next_sequence = 1
