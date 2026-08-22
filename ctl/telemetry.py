from collections import deque
from dataclasses import asdict
from threading import RLock
from typing import Any

from contracts import NodeMetrics
from ctl.registry import NodeRegistry


class TelemetryFanIn:
    def __init__(
        self,
        registry: NodeRegistry,
        *,
        interval_s: float = 0.5,
        history_limit: int = 60,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be greater than zero")
        if history_limit <= 0:
            raise ValueError("history_limit must be greater than zero")

        self.registry = registry
        self.interval_s = interval_s
        self.history_limit = history_limit

        self.latest: dict[str, NodeMetrics] = {}
        self.samples: dict[str, deque[NodeMetrics]] = {}
        self.llama_metrics: dict[str, float] = {}
        self.llama_samples: deque[dict[str, float]] = deque(maxlen=history_limit)
        self.poll_errors: dict[str, str] = {}
        self.lock = RLock()

    def record(self, metrics: NodeMetrics) -> None:
        with self.lock:
            history = self.samples.setdefault(
                metrics.node_id,
                deque(maxlen=self.history_limit),
            )
            history.append(metrics)
            self.latest[metrics.node_id] = metrics

    def reset(self, metrics: list[NodeMetrics] | None = None) -> None:
        with self.lock:
            self.latest.clear()
            self.samples.clear()
            self.llama_metrics.clear()
            self.llama_samples.clear()
            self.poll_errors.clear()

        for sample in metrics or []:
            self.record(sample)

    def remove(self, node_id: str) -> None:
        with self.lock:
            self.latest.pop(node_id, None)
            self.samples.pop(node_id, None)
            self.poll_errors.pop(node_id, None)

    def frame(self) -> dict[str, Any]:
        with self.lock:
            node_ids = sorted(self.latest)

            return {
                "type": "metrics",
                "nodes": [asdict(self.latest[node_id]) for node_id in node_ids],
                "history": {
                    node_id: [asdict(sample) for sample in self.samples[node_id]]
                    for node_id in node_ids
                },
                "llama": dict(self.llama_metrics),
                "llama_history": [dict(sample) for sample in self.llama_samples],
                "errors": dict(self.poll_errors),
            }
