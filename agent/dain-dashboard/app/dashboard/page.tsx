"use client";

// app/dashboard/page.tsx — live node view.
//
// This was a server component doing a one-shot `await getNodes()`. That had
// three problems: the numbers never moved after the initial render, ctl being
// down threw during render and 500'd the whole page instead of showing the
// empty state, and it bypassed the feed socket the rest of the app runs on.
// It now reads useNodes(), which pairs each node's static NodeProfile with its
// latest NodeMetrics sample and re-renders as telemetry arrives at 2 Hz.

import styles from "./page.module.css";
import {
  Cpu,
  MemoryStick,
  MonitorCog,
  Gauge,
  Network,
  Zap,
  Activity,
  Briefcase,
  Gpu,
  CircuitBoard,
  PcCase,
  SquareActivity,
} from "lucide-react";

import { useNodes, type LiveNode } from "../../lib/feed/useNodes";
import { useFeed } from "../../lib/feed/FeedProvider";

function formatMemory(mb: number | null | undefined) {
  if (mb === null || mb === undefined || !Number.isFinite(mb)) return "—";
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${Math.round(mb).toLocaleString()} MB`;
}

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  return `${value.toFixed(1)}%`;
}

function formatNumber(value: unknown, unit = "") {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value}${unit}`
    : "—";
}

/** Live free RAM when telemetry is flowing, else the node's join-time figure.
 *  Returns null when neither is usable so the bar renders empty rather than
 *  at NaN% — the old code read `node.ram_free`, a field that does not exist on
 *  NodeProfile (it is `ram_free_mb`), so every bar was NaN. */
function ramUsagePercent(node: LiveNode): number | null {
  const total = node.ram_total_mb as number | undefined;
  const free =
    node.metrics?.ram_free_mb ?? (node.ram_free_mb as number | undefined);
  if (!total || !Number.isFinite(total)) return null;
  if (free === undefined || !Number.isFinite(free)) return null;
  return Math.min(100, Math.max(0, ((total - free) / total) * 100));
}

function clampPercent(value: number | null | undefined): number {
  if (value === null || value === undefined || !Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, value));
}

export default function Dashboard() {
  const nodes = useNodes();
  const { connected } = useFeed();

  const rows = [...nodes].sort((a, b) => a.id.localeCompare(b.id));

  return (
    <main className={styles.stage}>
      <header className={styles.header}>
        <div>
          <p className={styles.kicker}>D.A.I.N</p>
          <h1 className={styles.title}>
            {rows.length > 0 ? "Available Nodes" : "No Available Nodes"}
          </h1>
        </div>
        <span className={styles.headerStatus} data-connected={connected}>
          <SquareActivity size={13} />
          {connected ? `${rows.length} live` : "feed offline"}
        </span>
      </header>

      {rows.length === 0 && (
        <p className={styles.muted}>
          {connected
            ? "Connected to the control plane, but no nodes have joined yet."
            : "Waiting for the control plane — retrying every 2s. Check that ctl is running and that NEXT_PUBLIC_FEED_URL points at it."}
        </p>
      )}

      <section className={styles.grid}>
        {rows.map((node) => {
          const metric = node.metrics;
          const totalRamMb = node.ram_total_mb as number | undefined;
          const freeRamMb =
            metric?.ram_free_mb ?? (node.ram_free_mb as number | undefined);
          const usedRamMb =
            totalRamMb !== undefined && freeRamMb !== undefined
              ? totalRamMb - freeRamMb
              : undefined;

          return (
            <article key={node.id} className={styles.card}>
              <div className={styles.cardTop}>
                <div className={styles.nodeIdentity}>
                  <span className={styles.nodeIcon}>
                    <PcCase size={17} /> 
                  </span>
                  <span className={styles.nodeId}>{node.id}</span>
                </div>
                <span
                  className={styles.dot}
                  data-state={node.state}
                  title={node.state}
                />
              </div>

              <p className={styles.hostname}>{node.host as string}</p>

              <div className={styles.stats}>
                <div className={styles.statRow}>
                  <span className={styles.statLabel}><Cpu size={14} />CPU</span>
                  <span className={styles.statValue}>
                    {(node.cpu as string) ?? "—"}
                    <span className={styles.muted}>
                      {formatNumber(node.cores)} cores
                    </span>
                  </span>
                </div>

                {/* Live utilisation. The old card had nowhere to put
                    cpu_percent at all, so telemetry that was arriving fine had
                    no way to show up. */}
                <div className={styles.statBlock}>
                  <div className={styles.statRow}>
                    <span className={styles.statLabel}>
                      <Activity size={14} />Load
                    </span>
                    <span className={styles.statValue}>
                      {formatPercent(metric?.cpu_percent)}
                      {metric === undefined && (
                        <span className={styles.muted}>no telemetry</span>
                      )}
                    </span>
                  </div>
                  <div className={styles.progressTrack}>
                    <div
                      className={styles.progressBar}
                      style={{ width: `${clampPercent(metric?.cpu_percent)}%` }}
                    />
                  </div>
                </div>

                <div className={styles.statBlock}>
                  <div className={styles.statRow}>
                    <span className={styles.statLabel}>
                      <MemoryStick size={14} />RAM
                    </span>
                    <span className={styles.statValue}>
                      {formatMemory(usedRamMb)}
                      <span className={styles.muted}>
                        / {formatMemory(totalRamMb)}
                      </span>
                    </span>
                  </div>
                  <div className={styles.progressTrack}>
                    <div
                      className={styles.progressBar}
                      style={{ width: `${clampPercent(ramUsagePercent(node))}%` }}
                    />
                  </div>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}>
                    <Gpu size={14} />GPU
                  </span>
                  <span className={styles.statValue}>
                    {node.gpu ? (
                      <>
                        {node.gpu as string}
                        <span className={styles.muted}>
                          {metric?.vram_free_mb != null
                            ? `${formatMemory(metric.vram_free_mb)} free / ${formatMemory(node.vram_total_mb as number)}`
                            : `${formatMemory(node.vram_total_mb as number)} VRAM`}
                          {metric?.gpu_percent != null
                            ? ` · ${formatPercent(metric.gpu_percent)}`
                            : ""}
                        </span>
                      </>
                    ) : (
                      <span className={styles.muted}>None</span>
                    )}
                  </span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}>
                    <Briefcase size={14} />Jobs
                  </span>
                  <span className={styles.statValue}>
                    {formatNumber(metric?.jobs_running ?? 0)}
                    <span className={styles.muted}>running</span>
                  </span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}><CircuitBoard size={14} />Backend</span>
                  <span className={styles.badge}>
                    {(node.backend as string) ?? "—"}
                  </span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}>
                    <Gauge size={14} />Bandwidth
                  </span>
                  <span className={styles.statValue}>
                    {formatNumber(node.mem_bandwidth_gbs, " GB/s")}
                  </span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}><Zap size={14} />tg / pp</span>
                  <span className={styles.statValue}>
                    {formatNumber(node.tg_tok_s)} / {formatNumber(node.pp_tok_s)}
                    <span className={styles.muted}>tok/s</span>
                  </span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}>
                    <Network size={14} />RTT
                  </span>
                  <span className={styles.statValue}>
                    {formatNumber(node.rtt_ms, " ms")}
                  </span>
                </div>
              </div>

              <div className={styles.state} data-state={node.state}>
                <span />
                {node.state}
              </div>
            </article>
          );
        })}
      </section>
    </main>
  );
}
