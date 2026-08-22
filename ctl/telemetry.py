import asyncio
import time
from collections import deque
from dataclasses import asdict
from threading import RLock
from typing import Any

import httpx

from contracts import NodeMetrics, NodeProfile
from ctl.registry import NodeRegistry


class TelemetryFanIn:
    def __init__(
        self,
        registry: NodeRegistry,
        *,
        interval_s: float = 0.5,
        timeout_s: float = 1.0,
        history_limit: int = 60,
        node_port: int = 9100,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be greater than zero")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        if history_limit <= 0:
            raise ValueError("history_limit must be greater than zero")

        self.registry = registry
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self.history_limit = history_limit
        self.node_port = node_port

        self.client = client
        self.owns_client = client is None
        self.task: asyncio.Task[None] | None = None

        self.latest: dict[str, NodeMetrics] = {}
        self.samples: dict[str, deque[NodeMetrics]] = {}
        self.llama_metrics: dict[str, float] = {}
        self.llama_samples: deque[dict[str, float]] = deque(maxlen=history_limit)
        self.poll_errors: dict[str, str] = {}
        self.lock = RLock()

    async def start(self) -> None:
        if self.task is not None and not self.task.done():
            return

        self._ensure_client()
        self.task = asyncio.create_task(
            self._run(),
            name="dain-telemetry-fan-in",
        )

    async def close(self) -> None:
        task = self.task
        self.task = None

        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        if self.owns_client and self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()

        while True:
            started_at = loop.time()
            await self.poll_once()

            elapsed = loop.time() - started_at
            delay = max(0.0, self.interval_s - elapsed)
            await asyncio.sleep(delay)

    async def poll_once(self) -> None:
        self._ensure_client()

        profiles = [
            profile
            for profile in self.registry.list_profiles()
            if self._should_poll(profile)
        ]

        results = await asyncio.gather(
            *(self._poll_node(profile) for profile in profiles),
            return_exceptions=True,
        )

        for profile, result in zip(
            profiles,
            results,
            strict=True,
        ):
            if isinstance(result, Exception):
                with self.lock:
                    self.poll_errors[profile.id] = f"{type(result).__name__}: {result}"
                continue

            self.record(result)

            with self.lock:
                self.poll_errors.pop(profile.id, None)

    def record(self, metrics: NodeMetrics) -> None:
        with self.lock:
            history = self.samples.setdefault(
                metrics.node_id,
                deque(maxlen=self.history_limit),
            )
            history.append(metrics)
            self.latest[metrics.node_id] = metrics

    def reset(
        self,
        metrics: list[NodeMetrics] | None = None,
    ) -> None:
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

    def _should_poll(self, profile: NodeProfile) -> bool:
        record = self.registry.get_record(profile.id)

        return (
            record is not None
            and record.heartbeat_required
            and profile.state != "offline"
        )

    async def _poll_node(
        self,
        profile: NodeProfile,
    ) -> NodeMetrics:
        assert self.client is not None

        response = await self.client.get(
            self._node_url(profile),
            timeout=self.timeout_s,
        )
        response.raise_for_status()

        prometheus = parse_prometheus(response.text)

        return NodeMetrics(
            node_id=profile.id,
            timestamp=time.time(),
            cpu_percent=_required_metric(
                prometheus,
                "dain_node_cpu_percent",
            ),
            ram_free_mb=int(
                _required_metric(
                    prometheus,
                    "dain_node_ram_free_mb",
                )
            ),
            gpu_percent=_optional_metric(
                prometheus,
                "dain_node_gpu_percent",
            ),
            vram_free_mb=_optional_integer_metric(
                prometheus,
                "dain_node_vram_free_mb",
            ),
            jobs_running=int(
                prometheus.get(
                    "dain_node_jobs_running",
                    0.0,
                )
            ),
        )

    def _node_url(self, profile: NodeProfile) -> str:
        host = profile.host.rstrip("/")

        if host.startswith(("http://", "https://")):
            base = host
        elif ":" in host and host.rsplit(":", 1)[1].isdigit():
            base = f"http://{host}"
        else:
            base = f"http://{host}:{self.node_port}"

        return f"{base}/metrics"

    def _ensure_client(self) -> None:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient()
            self.owns_client = True


def parse_prometheus(body: str) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for raw_line in body.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue

        name, raw_value = parts

        try:
            metrics[name] = float(raw_value)
        except ValueError:
            continue

    return metrics


def _required_metric(
    metrics: dict[str, float],
    name: str,
) -> float:
    try:
        return metrics[name]
    except KeyError as exc:
        raise ValueError(f"node metrics response is missing {name}") from exc


def _optional_metric(
    metrics: dict[str, float],
    name: str,
) -> float | None:
    return metrics.get(name)


def _optional_integer_metric(
    metrics: dict[str, float],
    name: str,
) -> int | None:
    value = metrics.get(name)
    return None if value is None else int(value)
