// lib/feed/types.ts
// Shapes sent over /feed. See ctl's send_feed() — this is the single
// source of truth for what a frame can look like on the wire.

export type NodeState = "joining" | "idle" | "computing" | "degraded" | "offline";

export type NodeInfo = {
  id: string;
  host: string;
  state: NodeState;
  [key: string]: unknown; // profile fields vary; consumers pick what they need
};

export type TopologyFrame = {
  type: "topology";
  nodes: NodeInfo[];
};

export type MetricsFrame = {
  type: "metrics"; // adjust if get_metrics() omits "type" — see FeedProvider note
  [key: string]: unknown;
};

export type RegistryEventFrame = {
  type: "event";
  source: "registry";
  sequence: number;
  [key: string]: unknown; // join / leave / heartbeat-miss fields
};

export type QueueEventFrame = {
  type: "event";
  source: "queue";
  sequence: number;
  event: "queued" | "started" | "completed" | "failed" | "cancelled";
  job_id: string;
  [key: string]: unknown;
};

export type FlowFrame = {
  type: "flow";
  source: "ctl";
  target: string; // node_id
  label: "dispatch" | "retry" | "reassign";
  sequence: number;
  job_id: string;
  [key: string]: unknown;
};

export type Frame =
  | TopologyFrame
  | MetricsFrame
  | RegistryEventFrame
  | QueueEventFrame
  | FlowFrame;

export type JobResult = {
  shards?: { shard_index?: number; [key: string]: unknown }[];
  errors?: { shard_index: number; error: string }[];
  [key: string]: unknown;
};

export type Job = {
  id: string;
  kind?: string;
  payload?: { prompt?: string; [key: string]: unknown };
  status?: string;
  node_id?: string | null;   // null for fanned-out jobs — see assigned_nodes
  assigned_nodes?: string[]; // the real destination list when node_id is null
  fanout?: number;
  result?: JobResult | null;
  started_at?: number;  // unix seconds
  finished_at?: number | null;
  lastFlow?: string;
  sequence?: number;
  _lastSeen: number;
};