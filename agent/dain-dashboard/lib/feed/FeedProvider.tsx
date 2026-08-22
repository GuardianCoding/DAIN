// lib/feed/FeedProvider.tsx
// Opens exactly ONE WebSocket to /feed for the whole app and derives
// long-lived state (nodes, metrics, jobs) directly in the provider, not
// per-hook.
//
// Why state lives here rather than in each hook: "topology" is sent ONCE,
// right when the socket opens. If we only exposed the latest raw frame,
// any component that mounts after that first frame (e.g. navigating to a
// page later) would never see it. Likewise job history would reset every
// time a page using useJobs unmounted. Keeping nodes/jobs as accumulated
// state in the provider means every consumer — no matter when it mounts —
// reads the current picture, not a stream it had to be present for.
//
// Metrics: ctl's TelemetryFanIn.frame() DOES tag itself "type": "metrics"
// (an earlier note here claimed it didn't). We keep only the latest sample
// per node and drop `history`, which re-sends up to 60 samples per node on
// every frame at 2 Hz and which nothing renders.

"use client";

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  ReactNode,
} from "react";
import { API_BASE, FEED_URL } from "../config";
import type { Frame, MetricsFrame, NodeInfo, NodeMetrics, Job } from "./types";

type FeedContextValue = {
  connected: boolean;
  nodes: NodeInfo[];
  metrics: Map<string, NodeMetrics>;
  jobs: Job[];
  seedJob: (job: Job) => void;
  refreshJob: (id: string) => Promise<void>;
};

const FeedContext = createContext<FeedContextValue>({
  connected: false,
  nodes: [],
  metrics: new Map(),
  jobs: [],
  seedJob: () => {},
  refreshJob: async () => {},
});

const RECONNECT_MS = 2000;
const STALE_MS = 10_000;

export function FeedProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false);
  const [nodes, setNodes] = useState<Map<string, NodeInfo>>(new Map());
  const [metrics, setMetrics] = useState<Map<string, NodeMetrics>>(new Map());
  const [jobs, setJobs] = useState<Map<string, Job>>(new Map());
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;

  const wsRef = useRef<WebSocket | null>(null);
  // Explicit `undefined` argument: React 19's types reject the zero-arg
  // overload of useRef, and `next build` typechecks where `next dev` does not.
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );

  function upsertJob(job_id: string, patch: Partial<Job>) {
    setJobs((prev) => {
      const next = new Map(prev);
      const existing = next.get(job_id) ?? ({ id: job_id } as Job);
      next.set(job_id, { ...existing, ...patch, _lastSeen: Date.now() });
      return next;
    });
  }

  async function refreshJob(id: string) {
    try {
      const res = await fetch(`${API_BASE}/jobs/${id}`);
      if (res.ok) {
        upsertJob(id, await res.json());
      } else {
        console.warn(`refreshJob(${id}) failed: ${res.status}`);
      }
    } catch (err) {
      console.warn(`refreshJob(${id}) network error`, err);
    }
  }

  function handleFrame(frame: Frame) {
    if (frame.type === "topology") {
      const next = new Map<string, NodeInfo>();
      for (const n of frame.nodes) next.set(n.id, n);
      setNodes(next);
      return;
    }

    if (frame.type === "metrics") {
      const samples = (frame as MetricsFrame).nodes;
      if (!Array.isArray(samples)) return;
      setMetrics((prev) => {
        const next = new Map(prev);
        for (const sample of samples) {
          if (sample?.node_id) next.set(sample.node_id, sample);
        }
        return next;
      });
      return;
    }

    if (frame.type === "event" && frame.source === "registry") {
      const id = (frame as any).node_id ?? (frame as any).id;
      if (!id) return;
      setNodes((prev) => {
        const next = new Map(prev);
        const existing = next.get(id) ?? ({ id } as NodeInfo);
        next.set(id, { ...existing, ...frame });
        return next;
      });
      return;
    }

    if (frame.type === "event" && frame.source === "queue") {
      const isNew = !jobsRef.current.has(frame.job_id);
      upsertJob(frame.job_id, { status: frame.event, ...frame });
      // Queue events don't carry kind/node_id — only the id + status change.
      // If we're seeing this job for the first time (e.g. it existed before
      // this page loaded, or was created from another tab), fetch its full
      // record once so Kind/Node aren't left permanently blank.
      if (isNew) refreshJob(frame.job_id);
      return;
    }

    if (frame.type === "flow" && frame.source === "ctl") {
      upsertJob(frame.job_id, {
        node_id: frame.target,
        lastFlow: frame.label,
        sequence: frame.sequence,
      });
      return;
    }
  }

  useEffect(() => {
    let cancelled = false;

    function scheduleReconnect() {
      if (cancelled) return;
      clearTimeout(reconnectRef.current);
      reconnectRef.current = setTimeout(connect, RECONNECT_MS);
    }

    function connect() {
      if (cancelled) return;

      let ws: WebSocket;
      try {
        ws = new WebSocket(FEED_URL);
      } catch (err) {
        // A malformed URL throws RIGHT HERE, synchronously, before any handler
        // could be attached — so without this catch there is no onclose to arm
        // the retry and the feed stays dead forever, silently. That is exactly
        // what an unset NEXT_PUBLIC_FEED_URL used to do; lib/config.ts now
        // supplies a default, and this keeps a typo'd override recoverable.
        console.error(`feed: cannot open WebSocket to "${FEED_URL}"`, err);
        setConnected(false);
        scheduleReconnect();
        return;
      }

      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) {
          ws.close(); // effect was cleaned up while we were connecting
          return;
        }
        setConnected(true);
      };

      ws.onmessage = (msg) => {
        if (cancelled) return;
        let frame: Frame;
        try {
          const raw = JSON.parse(msg.data);
          frame = raw.type ? raw : { type: "metrics", ...raw };
        } catch (err) {
          // One unparseable frame must not take down the socket.
          console.warn("feed: dropping unparseable frame", err);
          return;
        }
        handleFrame(frame);
      };

      // The browser always fires close after error, so the retry is armed
      // there; this exists so the reason reaches the console.
      ws.onerror = () => {
        if (!cancelled) console.warn(`feed: socket error on ${FEED_URL}`);
      };

      ws.onclose = () => {
        if (cancelled) return;
        setConnected(false);
        scheduleReconnect();
      };
    }

    connect();

    const staleInterval = setInterval(() => {
      const now = Date.now();
      for (const j of jobsRef.current.values()) {
        if (j.status === "running" && now - j._lastSeen > STALE_MS) {
          refreshJob(j.id);
        }
      }
    }, STALE_MS);

    return () => {
      cancelled = true;
      clearTimeout(reconnectRef.current);
      clearInterval(staleInterval);
      // Only close if the socket actually reached OPEN — closing a socket
      // still mid-handshake is exactly what produces the console warning.
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
    };
  }, []);

  function seedJob(job: Job) {
    upsertJob(job.id, job);
  }

  return (
    <FeedContext.Provider
      value={{
        connected,
        nodes: [...nodes.values()],
        metrics,
        jobs: [...jobs.values()],
        seedJob,
        refreshJob,
      }}
    >
      {children}
    </FeedContext.Provider>
  );
}

export function useFeed() {
  return useContext(FeedContext);
}
