// lib/feed/useNodes.ts
// Thin selector over FeedProvider's accumulated node state. All parsing and
// accumulation happens once, in the provider — this just reads it and pairs
// each node with its latest telemetry sample.
//
// The two halves mean different things and are deliberately NOT flattened
// into one another:
//
//   node.*         static NodeProfile, frozen at join — id, host, cpu model,
//                  cores, ram_total_mb, gpu, backend, bandwidth, tg/pp, rtt
//   node.metrics.* live NodeMetrics, resampled at 2 Hz — cpu_percent,
//                  ram_free_mb, gpu_percent, vram_free_mb, jobs_running
//
// Both carry a field called ram_free_mb. The profile's is a join-time number
// that never moves; the metric's is current. Merging them would silently pick
// one, so callers choose explicitly, e.g.
//   const freeRamMb = node.metrics?.ram_free_mb ?? node.ram_free_mb;
// `metrics` is undefined until the first metrics frame arrives, and for any
// node ctl cannot currently poll.

"use client";

import { useFeed } from "./FeedProvider";
import type { NodeInfo, NodeMetrics } from "./types";

export type LiveNode = NodeInfo & { metrics?: NodeMetrics };

export function useNodes(): LiveNode[] {
  const { nodes, metrics } = useFeed();
  return nodes.map((node) => ({ ...node, metrics: metrics.get(node.id) }));
}
