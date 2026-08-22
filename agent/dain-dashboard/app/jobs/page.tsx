"use client";

// app/jobs/page.tsx — pure tracking view. All feed handling lives in
// lib/feed/FeedProvider.tsx, consumed here via the useJobs() selector.
//
// Expanded row shows every field on the job record:
//   id, kind, payload, node_id, status, result (shards + errors),
//   started_at, finished_at, fanout, assigned_nodes

import { useState, Fragment } from "react";
import styles from "./page.module.css";
import { useJobs } from "../../lib/feed/useJobs";
import {
  Loader2, CheckCircle2, XCircle, Clock, CircleDot, ChevronDown, ChevronRight,
} from "lucide-react";
import type { Job } from "../../lib/feed/types";

const STATUS_ICON: Record<string, typeof CircleDot> = {
  queued: Clock,
  running: Loader2,
  done: CheckCircle2,
  failed: XCircle,
  cancelled: XCircle,
};

function formatDuration(job: Job) {
  if (!job.started_at) return "—";
  const end = job.finished_at ?? Date.now() / 1000;
  const secs = end - job.started_at;
  if (secs < 1) return "<1s";
  if (secs < 60) return `${secs.toFixed(1)}s`;
  return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
}

function formatTimestamp(unixSeconds?: number | null) {
  if (!unixSeconds) return "—";
  const d = new Date(unixSeconds * 1000);
  return `${d.toLocaleTimeString()} · ${d.toLocaleDateString()}`;
}

export default function JobsPage() {
  const { jobs } = useJobs();
  const rows = [...jobs].sort((a, b) => b._lastSeen - a._lastSeen);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  function toggle(id: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  return (
    <main className={styles.stage}>
      <header className={styles.header}>
        <div>
          <p className={styles.kicker}>D.A.I.N</p>
          <h1 className={styles.title}>Jobs</h1>
        </div>
        <span className={styles.badge}>{rows.length} tracked</span>
      </header>

      <div className={styles.card}>
        {rows.length === 0 ? (
          <p className={styles.muted}>No jobs yet — submit one from Create Job.</p>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th style={{ width: 20 }}></th>
                <th>Job</th>
                <th>Kind</th>
                <th>Node(s)</th>
                <th>Status</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((j) => {
                const kind = j.kind || "—";
                const node = j.node_id
                  ? j.node_id
                  : j.assigned_nodes?.length
                  ? `${j.assigned_nodes.length} nodes`
                  : "unassigned";
                const status = j.status || "unknown";
                const Icon = STATUS_ICON[status] ?? CircleDot;
                const isOpen = expanded.has(j.id);
                const errors = j.result?.errors ?? [];
                const shards = j.result?.shards ?? [];

                return (
                  <Fragment key={j.id}>
                    <tr className={styles.row} onClick={() => toggle(j.id)}>
                      <td>{isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</td>
                      <td className={styles.mono}>{j.id.slice(0, 8)}</td>
                      <td>{kind}</td>
                      <td>{node}</td>
                      <td className={styles.statusCell} data-status={status}>
                        <Icon size={14} className={status === "running" ? styles.spin : undefined} />
                        {status}
                        {errors.length > 0 && (
                          <span className={styles.errorBadge}>
                            {errors.length} error{errors.length > 1 ? "s" : ""}
                          </span>
                        )}
                      </td>
                      <td>{formatDuration(j)}</td>
                    </tr>

                    {isOpen && (
                      <tr className={styles.detailRow}>
                        <td colSpan={6}>
                          <div className={styles.detailGrid}>

                            <div>
                              <span className={styles.detailLabel}>Full ID</span>
                              <p className={styles.mono}>{j.id}</p>
                            </div>
                            <div>
                              <span className={styles.detailLabel}>Kind</span>
                              <p>{j.kind ?? "—"}</p>
                            </div>
                            <div>
                              <span className={styles.detailLabel}>Node ID</span>
                              <p className={styles.mono}>
                                {j.node_id ?? "null (fanned out)"}
                              </p>
                            </div>
                            <div>
                              <span className={styles.detailLabel}>Fanout</span>
                              <p>{j.fanout ?? "—"}</p>
                            </div>

                            <div>
                              <span className={styles.detailLabel}>Started</span>
                              <p>{formatTimestamp(j.started_at)}</p>
                            </div>
                            <div>
                              <span className={styles.detailLabel}>Finished</span>
                              <p>{formatTimestamp(j.finished_at)}</p>
                            </div>
                            <div>
                              <span className={styles.detailLabel}>Duration</span>
                              <p>{formatDuration(j)}</p>
                            </div>
                            <div>
                              <span className={styles.detailLabel}>Shards</span>
                              <p>{shards.length} returned</p>
                            </div>

                            <div className={styles.detailFull}>
                              <span className={styles.detailLabel}>Assigned nodes</span>
                              <div className={styles.chipRow}>
                                {j.assigned_nodes?.length ? (
                                  j.assigned_nodes.map((n) => (
                                    <span key={n} className={styles.chip}>{n}</span>
                                  ))
                                ) : (
                                  <p className={styles.mutedInline}>none</p>
                                )}
                              </div>
                            </div>

                            <div className={styles.detailFull}>
                              <span className={styles.detailLabel}>Payload</span>
                              {j.payload?.prompt ? (
                                <p>{j.payload.prompt}</p>
                              ) : j.payload ? (
                                <pre className={styles.pre}>{JSON.stringify(j.payload, null, 2)}</pre>
                              ) : (
                                <p className={styles.mutedInline}>none</p>
                              )}
                            </div>

                            {shards.length > 0 && (
                              <div className={styles.detailFull}>
                                <span className={styles.detailLabel}>Shard results</span>
                                <pre className={styles.pre}>{JSON.stringify(shards, null, 2)}</pre>
                              </div>
                            )}

                            {errors.length > 0 && (
                              <div className={styles.detailFull}>
                                <span className={styles.detailLabel}>Errors</span>
                                <ul className={styles.errorList}>
                                  {errors.map((e, i) => (
                                    <li key={i}>
                                      <span className={styles.mono}>shard {e.shard_index}</span>
                                      {" — "}
                                      {e.error || "no message"}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </main>
  );
}