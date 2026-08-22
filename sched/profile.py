# sched/profile.py
"""Fetches the raw cluster state plan() needs: static NodeProfiles
(hardware/capability, mostly fixed after join) and live NodeMetrics
(current free memory/CPU/GPU load, refreshed every call).

This is the dev/test path — hitting Abdallah's mock (or the real control
plane) over HTTP so plan() can be exercised standalone. In the actual
demo, the control plane likely calls plan() in-process and passes these
same two lists directly, skipping the network hop entirely. Keep this
fetcher decoupled from plan() itself for exactly that reason.
"""

import httpx

from contracts import NodeMetrics, NodeProfile


class ClusterStateFetcher:
    def __init__(self, ctl_base_url: str, timeout: float = 5.0):
        self.ctl_base_url = ctl_base_url.rstrip("/")
        self.timeout = timeout

    async def get_profiles(self) -> list[NodeProfile]:
        """GET /api/nodes — static-ish hardware/capability data.
        Populated once per node at profiling/join time (SCH-1),
        NOT refreshed on every plan() call."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.ctl_base_url}/api/nodes")
            resp.raise_for_status()
            return [self._to_node_profile(n) for n in resp.json()]

    async def get_metrics(self) -> list[NodeMetrics]:
        """GET /api/metrics — live load/free-memory snapshot.
        This is what cost.fits() should check against, since it reflects
        what's actually free right now, not what was free at join time."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.ctl_base_url}/api/metrics")
            resp.raise_for_status()
            return [self._to_node_metrics(m) for m in resp.json()["nodes"]]

    async def get_cluster_state(self) -> tuple[list[NodeProfile], list[NodeMetrics]]:
        """Convenience: both lists together, since plan() needs both.
        Fetched as two separate calls, not one endpoint — they come from
        different sources at different refresh rates."""
        profiles = await self.get_profiles()
        metrics = await self.get_metrics()
        return profiles, metrics

    def _to_node_profile(self, data: dict) -> NodeProfile:
        return NodeProfile(
            id=data["id"],
            host=data["host"],
            cpu=data["cpu"],
            cores=data["cores"],
            ram_total_mb=data["ram_total_mb"],
            ram_free_mb=data["ram_free_mb"],
            gpu=data.get("gpu"),
            vram_total_mb=data["vram_total_mb"],
            backend=data["backend"],  # "cuda" | "cpu" | "vulkan" | "metal"
            mem_bandwidth_gbs=data["mem_bandwidth_gbs"],
            tg_tok_s=data["tg_tok_s"],
            pp_tok_s=data["pp_tok_s"],
            rtt_ms=data["rtt_ms"],
            state=data.get("state", "idle"),
        )

    def _to_node_metrics(self, data: dict) -> NodeMetrics:
        return NodeMetrics(
            node_id=data["node_id"],
            timestamp=data["timestamp"],
            cpu_percent=data["cpu_percent"],
            ram_free_mb=data["ram_free_mb"],
            gpu_percent=data.get("gpu_percent"),
            vram_free_mb=data.get("vram_free_mb"),
            jobs_running=data.get("jobs_running", 0),
        )
