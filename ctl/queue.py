import asyncio
import time
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

import httpx

from contracts import Job, NodeProfile
from ctl.registry import NodeRegistry
from node.auth import sign_job_request
from node.bench import BENCH_TIMEOUT_S
from node.infer import READY_TIMEOUT_S as INFER_READY_TIMEOUT_S
from node.infer import REQUEST_TIMEOUT_S as INFER_REQUEST_TIMEOUT_S
from node.sandbox import MAX_TIMEOUT_S as SANDBOX_MAX_TIMEOUT_S

JobKind = Literal["infer", "exec", "index", "search", "bench"]

DEFAULT_ENDPOINTS: dict[JobKind, str] = {
    "infer": "/infer",
    "exec": "/exec",
    "index": "/index",
    "search": "/search",
    "bench": "/bench",
}

# How much longer than the node's own ceiling ctl waits before giving up.
TIMEOUT_MARGIN_S = 60.0

# A dispatch timeout must EXCEED the node's own ceiling for that kind.
#
# Undercut it and the failure is not merely a slow job: a ReadTimeout is an
# httpx.HTTPError, so _run_shard cannot tell "still working" from "dead". It
# retries, then walks _ranked_nodes() reassigning the same payload to every
# remaining node. One 30-second generation becomes the same prompt running on
# the whole pool at once, returning nothing. Every kind except index used to
# share the 2.0s default, so /infer and /bench could never complete here.
#
# Each bound is derived from the node-side constant rather than written out,
# so retuning one of those cannot silently reintroduce the gap. The node's own
# timeouts already produce readable 503s ("llama-server exited with code ...")
# and those should win; this is only the backstop for a node that has stopped
# answering entirely.
#
# Every kind has an entry so the table is the single complete answer to "how
# long may this kind take?" — agent/client.py derives its own polling deadline
# from it, and a missing key there would silently mean "give up immediately".
DEFAULT_TIMEOUTS_S: dict[JobKind, float] = {
    # await_ready() polls for model load, then the generation itself.
    "infer": INFER_READY_TIMEOUT_S + INFER_REQUEST_TIMEOUT_S + TIMEOUT_MARGIN_S,
    # llama-bench -r 3 on a slow CPU node.
    "bench": BENCH_TIMEOUT_S + TIMEOUT_MARGIN_S,
    # The sandbox refuses a per-command timeout above its own maximum.
    "exec": SANDBOX_MAX_TIMEOUT_S + TIMEOUT_MARGIN_S,
    # Walking a disk and embedding every file. Overridden by index_timeout_s.
    "index": 30.0,
    # One query embedding against an index already in memory.
    "search": 30.0,
}


class NodeUnavailableError(RuntimeError):
    pass


class ShardExecutionError(RuntimeError):
    pass


class NodeRejectedError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueueEvent:
    sequence: int
    timestamp: float
    event: str
    job_id: str
    node_id: str | None
    status: str
    message: str


class JobQueue:
    def __init__(
        self,
        registry: NodeRegistry,
        *,
        pool_secret: str,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 2.0,
        index_timeout_s: float = 30.0,
        per_node_limit: int = 1,
        node_port: int = 9100,
        endpoint_by_kind: dict[JobKind, str] | None = None,
        timeout_by_kind: dict[JobKind, float] | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        if index_timeout_s <= 0:
            raise ValueError("index_timeout_s must be greater than zero")
        if per_node_limit <= 0:
            raise ValueError("per_node_limit must be greater than zero")
        if not pool_secret:
            raise ValueError("pool_secret must not be empty")

        self.registry = registry
        self.pool_secret = pool_secret
        self.timeout_s = timeout_s
        self.index_timeout_s = index_timeout_s
        self.per_node_limit = per_node_limit
        self.node_port = node_port
        self.endpoint_by_kind = endpoint_by_kind or DEFAULT_ENDPOINTS.copy()
        # index_timeout_s predates the table and stays the named knob for that
        # one kind; an explicit timeout_by_kind wins over both.
        self.timeout_by_kind: dict[JobKind, float] = {
            **DEFAULT_TIMEOUTS_S,
            "index": index_timeout_s,
            **(timeout_by_kind or {}),
        }

        self.client = client or httpx.AsyncClient(timeout=timeout_s)
        self.owns_client = client is None

        self.jobs: dict[str, Job] = {}
        self.fanouts: dict[str, int] = {}
        self.assigned_nodes: dict[str, list[str]] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.semaphores: dict[str, asyncio.Semaphore] = {}
        self.in_flight: dict[str, int] = {}

        self.events: list[QueueEvent] = []
        self.next_sequence = 1
        self.lock = RLock()
        self.closed = False

    async def submit(
        self,
        kind: JobKind,
        payload: dict[str, Any],
        fanout: int = 1,
        node_id: str | None = None,
    ) -> Job:
        if self.closed:
            raise RuntimeError("job queue is closed")
        if kind not in self.endpoint_by_kind:
            raise ValueError(f"unsupported job kind: {kind}")
        if fanout < 1:
            raise ValueError("fanout must be at least one")

        shards = self._split_payload(payload, 1 if node_id else fanout)
        if node_id is not None:
            self._profile(node_id)
            selected = [node_id]
        else:
            ranked = self._ranked_nodes()
            if len(ranked) < len(shards):
                raise RuntimeError(
                    f"fan-out {len(shards)} requested but only "
                    f"{len(ranked)} node(s) are available"
                )
            selected = ranked[: len(shards)]

        job = Job(
            id=uuid4().hex,
            kind=kind,
            payload=dict(payload),
            node_id=node_id,
        )

        with self.lock:
            self.jobs[job.id] = job
            self.fanouts[job.id] = len(shards)
            self.assigned_nodes[job.id] = list(selected)
            self._emit(
                "queued",
                job,
                None,
                f"Job {job.id} queued with fan-out {len(shards)}",
            )
            task = asyncio.create_task(
                self._run_job(job, shards, selected),
                name=f"dain-job-{job.id}",
            )
            self.tasks[job.id] = task
            task.add_done_callback(
                lambda done, job_id=job.id: self._task_finished(job_id, done)
            )

        return job

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def response(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return {
                **asdict(job),
                "fanout": self.fanouts[job_id],
                "assigned_nodes": list(self.assigned_nodes[job_id]),
            }

    def events_after(self, sequence: int) -> list[QueueEvent]:
        with self.lock:
            return [event for event in self.events if event.sequence > sequence]

    async def wait(self, job_id: str, timeout: float | None = None) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
            task = self.tasks.get(job_id)

        if job is None:
            raise KeyError(job_id)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout)
        return job

    async def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            tasks = list(self.tasks.values())

        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self.owns_client:
            await self.client.aclose()

    async def _run_job(
        self,
        job: Job,
        shards: list[dict[str, Any]],
        selected: list[str],
    ) -> None:
        with self.lock:
            job.status = "running"
            job.started_at = time.time()
            self._emit("started", job, None, f"Job {job.id} started")

        try:
            outcomes = await asyncio.gather(
                *(
                    self._run_shard(job, index, len(shards), shard, selected[index])
                    for index, shard in enumerate(shards)
                ),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            with self.lock:
                job.status = "failed"
                job.finished_at = time.time()
                job.result = {"shards": [], "errors": [{"error": "cancelled"}]}
                self._emit("cancelled", job, None, f"Job {job.id} was cancelled")
            raise

        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for index, outcome in enumerate(outcomes):
            if isinstance(outcome, BaseException):
                errors.append({"shard_index": index, "error": str(outcome)})
            else:
                results.append(outcome)

        with self.lock:
            job.result = {"shards": results, "errors": errors}
            if job.kind == "search":
                try:
                    job.result.update(self._merge_search_results(job, results))
                except ShardExecutionError as exc:
                    errors.append({"shard_index": None, "error": str(exc)})
            job.finished_at = time.time()
            job.status = "failed" if errors else "done"
            event = "failed" if errors else "completed"
            message = f"Job {job.id} {job.status}"
            if errors:
                message = f"{message}: {errors[0]['error']}"
            self._emit(event, job, None, message)

    async def _run_shard(
        self,
        job: Job,
        shard_index: int,
        shard_count: int,
        payload: dict[str, Any],
        initial_node: str,
    ) -> dict[str, Any]:
        attempted: set[str] = set()
        node_id = initial_node
        last_error: Exception | None = None

        while True:
            for attempt in range(2):
                try:
                    result = await self._request(
                        job, node_id, shard_index, shard_count, payload
                    )
                    return {
                        "shard_index": shard_index,
                        "node_id": node_id,
                        "result": result,
                    }
                except NodeRejectedError as exc:
                    raise ShardExecutionError(
                        f"shard {shard_index} rejected by {node_id}: {exc}"
                    ) from exc
                except (httpx.HTTPError, NodeUnavailableError, ValueError) as exc:
                    last_error = exc
                    if attempt == 0:
                        with self.lock:
                            self._emit(
                                "retrying",
                                job,
                                node_id,
                                f"Retrying shard {shard_index} on {node_id}",
                            )

            attempted.add(node_id)
            replacements = self._ranked_nodes(exclude=attempted)
            if not replacements:
                break

            previous = node_id
            node_id = replacements[0]
            with self.lock:
                if node_id not in self.assigned_nodes[job.id]:
                    self.assigned_nodes[job.id].append(node_id)
                self._emit(
                    "reassigned",
                    job,
                    node_id,
                    f"Shard {shard_index} moved from {previous} to {node_id}",
                )

        raise ShardExecutionError(
            f"shard {shard_index} exhausted available nodes: {last_error}"
        )

    async def _request(
        self,
        job: Job,
        node_id: str,
        shard_index: int,
        shard_count: int,
        payload: dict[str, Any],
    ) -> Any:
        semaphore = self._semaphore(node_id)
        async with semaphore:
            profile = self._profile(node_id)
            with self.lock:
                self.in_flight[node_id] = self.in_flight.get(node_id, 0) + 1
                self._emit(
                    "dispatched",
                    job,
                    node_id,
                    f"Shard {shard_index} sent to {node_id}",
                )

            try:
                issued_at = int(time.time())
                request_body = {
                    "job_id": job.id,
                    "kind": job.kind,
                    "payload": payload,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "issued_at": issued_at,
                }
                request_body["signature"] = sign_job_request(
                    self.pool_secret,
                    job_id=job.id,
                    kind=job.kind,
                    payload=payload,
                    shard_index=shard_index,
                    shard_count=shard_count,
                    issued_at=issued_at,
                )
                response = await self.client.post(
                    self._node_url(profile, self.endpoint_by_kind[job.kind]),
                    json=request_body,
                    timeout=self.timeout_by_kind.get(job.kind, self.timeout_s),
                )
                if 400 <= response.status_code < 500:
                    try:
                        rejection = response.json()
                    except ValueError:
                        detail = response.text
                    else:
                        detail = (
                            rejection.get("detail", response.text)
                            if isinstance(rejection, dict)
                            else response.text
                        )
                    raise NodeRejectedError(f"HTTP {response.status_code}: {detail}")
                response.raise_for_status()
                body = response.json()
                if isinstance(body, dict) and body.get("ok") is False:
                    raise ValueError(body.get("error", "node rejected the job"))
                if isinstance(body, dict) and "result" in body:
                    return body["result"]
                return body
            finally:
                with self.lock:
                    self.in_flight[node_id] = max(0, self.in_flight.get(node_id, 1) - 1)

    def _merge_search_results(
        self,
        job: Job,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        merged_hits: list[dict[str, Any]] = []
        nodes_searched: set[str] = set()
        embedding_models: set[str] = set()

        for shard in results:
            node_id = shard["node_id"]
            nodes_searched.add(node_id)
            result = shard.get("result")
            if not isinstance(result, dict):
                continue
            model_id = result.get("embedding_model")
            if isinstance(model_id, str) and model_id:
                embedding_models.add(model_id)

            hits = result.get("hits", [])
            if not isinstance(hits, list):
                continue

            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                path = hit.get("path")
                score = hit.get("score")
                if not isinstance(path, str) or not isinstance(score, (int, float)):
                    continue

                merged_hits.append(
                    {
                        **hit,
                        "node_id": node_id,
                        "source": f"{node_id}:{path}",
                        "shard_index": shard["shard_index"],
                    }
                )

        merged_hits.sort(
            key=lambda hit: (
                -hit["score"],
                hit["node_id"],
                hit["path"],
            )
        )
        requested_limit = job.payload.get("limit", 5)
        limit = (
            requested_limit
            if isinstance(requested_limit, int)
            and not isinstance(requested_limit, bool)
            and requested_limit > 0
            else 5
        )
        if len(embedding_models) > 1:
            raise ShardExecutionError(
                "search nodes returned different embedding models: "
                + ", ".join(sorted(embedding_models))
            )

        merged = {
            "hits": merged_hits[:limit],
            "nodes_searched": sorted(nodes_searched),
        }
        if embedding_models:
            merged["embedding_model"] = next(iter(embedding_models))
        return merged

    def _ranked_nodes(self, exclude: set[str] | None = None) -> list[str]:
        excluded = exclude or set()
        profiles = [
            profile
            for profile in self.registry.list_profiles()
            if profile.id not in excluded
            and profile.state not in {"joining", "offline"}
        ]
        metrics = {metric.node_id: metric for metric in self.registry.latest_metrics()}
        with self.lock:
            profiles.sort(
                key=lambda profile: (
                    (metrics[profile.id].jobs_running if profile.id in metrics else 0)
                    + self.in_flight.get(profile.id, 0),
                    profile.id,
                )
            )
        return [profile.id for profile in profiles]

    def _profile(self, node_id: str) -> NodeProfile:
        record = self.registry.get_record(node_id)
        if record is None:
            raise NodeUnavailableError(f"node {node_id} is not registered")
        if record.profile.state in {"joining", "offline"}:
            raise NodeUnavailableError(f"node {node_id} is {record.profile.state}")
        return record.profile

    def _split_payload(
        self, payload: dict[str, Any], fanout: int
    ) -> list[dict[str, Any]]:
        for key in ("tasks", "items"):
            values = payload.get(key)
            if isinstance(values, list) and values:
                shard_count = min(fanout, len(values))
                chunks: list[list[Any]] = [[] for _ in range(shard_count)]
                for index, value in enumerate(values):
                    chunks[index % shard_count].append(value)
                return [{**payload, key: chunk} for chunk in chunks]
        return [dict(payload) for _ in range(fanout)]

    def _semaphore(self, node_id: str) -> asyncio.Semaphore:
        with self.lock:
            if node_id not in self.semaphores:
                self.semaphores[node_id] = asyncio.Semaphore(self.per_node_limit)
            return self.semaphores[node_id]

    def _node_url(self, profile: NodeProfile, endpoint: str) -> str:
        host = profile.host.rstrip("/")
        if host.startswith(("http://", "https://")):
            base = host
        elif ":" in host and host.rsplit(":", 1)[1].isdigit():
            base = f"http://{host}"
        else:
            base = f"http://{host}:{self.node_port}"
        return f"{base}{endpoint}"

    def _emit(
        self,
        event: str,
        job: Job,
        node_id: str | None,
        message: str,
    ) -> None:
        self.events.append(
            QueueEvent(
                sequence=self.next_sequence,
                timestamp=time.time(),
                event=event,
                job_id=job.id,
                node_id=node_id,
                status=job.status,
                message=message,
            )
        )
        self.next_sequence += 1

    def _task_finished(self, job_id: str, finished: asyncio.Task[None]) -> None:
        with self.lock:
            if self.tasks.get(job_id) is finished:
                self.tasks.pop(job_id, None)

            if finished.cancelled():
                return
            error = finished.exception()
            if error is not None:
                job = self.jobs[job_id]
                job.status = "failed"
                job.finished_at = time.time()
                job.result = {"shards": [], "errors": [{"error": str(error)}]}
                self._emit("failed", job, None, f"Job {job.id} failed: {error}")
