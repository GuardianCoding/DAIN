// lib/feed/FeedProvider.tsx
// Opens exactly ONE WebSocket to /feed for the whole app and derives
// long-lived state (nodes, jobs) directly in the provider, not per-hook.
//
// Why state lives here rather than in each hook: "topology" is sent ONCE,
// right when the socket opens. If we only exposed the latest raw frame,
// any component that mounts after that first frame (e.g. navigating to a
// page later) would never see it. Likewise job history would reset every
// time a page using useJobs unmounted. Keeping nodes/jobs as accumulated
// state in the provider means every consumer — no matter when it mounts —
// reads the current picture, not a stream it had to be present for.
//
// NOTE on metrics frames: get_metrics() is sent as
// `await websocket.send_json(get_metrics())` with no "type" key added,
// unlike every other frame. We tag it "metrics" when "type" is absent —
// confirm this holds for your actual get_metrics() shape.

"use client";

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  ReactNode,
} from "react";
import type { Frame, NodeInfo, Job } from "./types";

type FeedContextValue = {
  connected: boolean;
  nodes: NodeInfo[];
  jobs: Job[];
  seedJob: (job: Job) => void;
  refreshJob: (id: string) => Promise<void>;
};

const FeedContext = createContext<FeedContextValue>({
  connected: false,
  nodes: [],
  jobs: [],
  seedJob: () => {},
  refreshJob: async () => {},
});

const FEED_URL =
  process.env.NEXT_PUBLIC_FEED_URL;
// Same host as the feed, but http(s) — used for the one-off REST calls this
// provider makes (job hydration). Relative fetch("/api/...") would instead
// hit the Next.js app's own origin, not the ctl backend, and 404 silently.
const API_BASE =
  process.env.NEXT_PUBLIC_API_URL;
const RECONNECT_MS = 2000;
const STALE_MS = 10_000;

export function FeedProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false);
  const [nodes, setNodes] = useState<Map<string, NodeInfo>>(new Map());
  const [jobs, setJobs] = useState<Map<string, Job>>(new Map());
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>();

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
    // metrics frames: not accumulated here — add a `metrics` state if a
    // page needs live numbers, following the same pattern as nodes/jobs.
  }

  useEffect(() => {
    let cancelled = false;

    function connect() {
      if (cancelled) return;
      const ws = new WebSocket(FEED_URL);
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
        const raw = JSON.parse(msg.data);
        const frame: Frame = raw.type ? raw : { type: "metrics", ...raw };
        handleFrame(frame);
      };

      ws.onclose = () => {
        if (cancelled) return;
        setConnected(false);
        reconnectRef.current = setTimeout(connect, RECONNECT_MS);
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
