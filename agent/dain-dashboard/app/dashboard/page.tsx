import styles from "./page.module.css";
import {
  Cpu,
  MemoryStick,
  MonitorCog,
  Gauge,
  Network,
  Zap,
  Activity,
} from "lucide-react";

import { getNodes } from "../components/API/api";

function formatMemory(mb: number) {
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }
  return `${mb.toLocaleString()} MB`;
}

function getRamUsage(node: any) {
  return ((node.ram_total_mb - node.ram_free) / node.ram_total_mb) * 100;
}

export default async function Dashboard() {
  const nodes = await getNodes();

  if (!nodes) {
    return (
      <main className={styles.stage}>
      <header className={styles.header}>
        <div>
          <p className={styles.kicker}>D.A.I.N</p>
          <h1 className={styles.title}>No Available Nodes.</h1>
        </div>
      </header>
      </main>
    )
  }

  else return (
    <main className={styles.stage}>
      <header className={styles.header}>
        <div>
          <p className={styles.kicker}>D.A.I.N</p>
          <h1 className={styles.title}>Available Nodes</h1>
        </div>
      </header>

      <section className={styles.grid}>
        {nodes.map((node) => {
          const ramUsage = getRamUsage(node);

          return (
            <article key={node.id} className={styles.card}>
              <div className={styles.cardTop}>
                <div className={styles.nodeIdentity}>
                  <span className={styles.nodeIcon}>
                    {node.gpu ? <MonitorCog size={17} /> : <Cpu size={17} />}
                  </span>
                  <span className={styles.nodeId}>{node.id}</span>
                </div>
                <span className={styles.dot} data-state={node.state} title={node.state} />
              </div>

              <p className={styles.hostname}>{node.host}</p>

              <div className={styles.stats}>
                <div className={styles.statRow}>
                  <span className={styles.statLabel}><Cpu size={14} />CPU</span>
                  <span className={styles.statValue}>
                    {node.cpu}
                    <span className={styles.muted}>{node.cores} cores</span>
                  </span>
                </div>

                <div className={styles.statBlock}>
                  <div className={styles.statRow}>
                    <span className={styles.statLabel}><MemoryStick size={14} />RAM</span>
                    <span className={styles.statValue}>
                      {formatMemory(node.ram_total_mb - node.ram_free_mb)}
                      <span className={styles.muted}>/ {formatMemory(node.ram_total_mb)}</span>
                    </span>
                  </div>
                  <div className={styles.progressTrack}>
                    <div className={styles.progressBar} style={{ width: `${ramUsage}%` }} />
                  </div>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}><MonitorCog size={14} />GPU</span>
                  <span className={styles.statValue}>
                    {node.gpu ? (
                      <>
                        {node.gpu}
                        <span className={styles.muted}>{formatMemory(node.vram_total_mb)} VRAM</span>
                      </>
                    ) : (
                      <span className={styles.muted}>None</span>
                    )}
                  </span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}><Zap size={14} />Backend</span>
                  <span className={styles.badge}>{node.backend}</span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}><Gauge size={14} />Bandwidth</span>
                  <span className={styles.statValue}>{node.mem_bandwidth_gbs} GB/s</span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}><Zap size={14} />tg / pp</span>
                  <span className={styles.statValue}>
                    {node.tg_tok_s} / {node.pp_tok_s}
                    <span className={styles.muted}>tok/s</span>
                  </span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}><Network size={14} />RTT</span>
                  <span className={styles.statValue}>{node.rtt_ms} ms</span>
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