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

// Stand-in data — replace with the real /api/nodes feed later.
const nodes = [
  {
    id: "gpu-01",
    hostname: "youssef-desktop",
    cpu_name: "Intel i5-12400",
    num_cores: 6,
    ram_total: 32000,
    ram_free: 18400,
    gpu: "RTX 3060",
    vram_total: 12000,
    backend: "vulkan",
    mem_bandwidth: 45.2,
    tg_tok_s: 124.5,
    pp_tok_s: 610.2,
    rtt_ms: 0.3,
    state: "joined",
  },
  {
    id: "office-01",
    hostname: "office-pc-01",
    cpu_name: "Intel i5-9500",
    num_cores: 6,
    ram_total: 16000,
    ram_free: 9800,
    gpu: null,
    vram_total: 0,
    backend: "cpu",
    mem_bandwidth: 22.1,
    tg_tok_s: 14.7,
    pp_tok_s: 78.4,
    rtt_ms: 0.4,
    state: "joined",
  },
  
];

function formatMemory(mb: number) {
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }

  return `${mb.toLocaleString()} MB`;
}

function getRamUsage(node: (typeof nodes)[number]) {
  return ((node.ram_total - node.ram_free) / node.ram_total) * 100;
}

export default function Dashboard() {
  return (
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
                    {node.gpu ? (
                      <MonitorCog size={17} />
                    ) : (
                      <Cpu size={17} />
                    )}
                  </span>

                  <span className={styles.nodeId}>{node.id}</span>
                </div>

                <span
                  className={styles.dot}
                  data-state={node.state}
                  title={node.state}
                />
              </div>

              <p className={styles.hostname}>{node.hostname}</p>

              <div className={styles.stats}>
                <div className={styles.statRow}>
                  <span className={styles.statLabel}>
                    <Cpu size={14} />
                    CPU
                  </span>

                  <span className={styles.statValue}>
                    {node.cpu_name}
                    <span className={styles.muted}>
                      {node.num_cores}c
                    </span>
                  </span>
                </div>

                <div className={styles.statBlock}>
                  <div className={styles.statRow}>
                    <span className={styles.statLabel}>
                      <MemoryStick size={14} />
                      RAM
                    </span>

                    <span className={styles.statValue}>
                      {formatMemory(node.ram_total - node.ram_free)}
                      <span className={styles.muted}>
                        / {formatMemory(node.ram_total)}
                      </span>
                    </span>
                  </div>

                  <div className={styles.progressTrack}>
                    <div
                      className={styles.progressBar}
                      style={{ width: `${ramUsage}%` }}
                    />
                  </div>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}>
                    <MonitorCog size={14} />
                    GPU
                  </span>

                  <span className={styles.statValue}>
                    {node.gpu ? (
                      <>
                        {node.gpu}
                        <span className={styles.muted}>
                          {formatMemory(node.vram_total)} VRAM
                        </span>
                      </>
                    ) : (
                      <span className={styles.muted}>None</span>
                    )}
                  </span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}>
                    <Zap size={14} />
                    Backend
                  </span>

                  <span className={styles.badge}>
                    {node.backend}
                  </span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}>
                    <Gauge size={14} />
                    Bandwidth
                  </span>

                  <span className={styles.statValue}>
                    {node.mem_bandwidth} GB/s
                  </span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}>
                    <Zap size={14} />
                    tg / pp
                  </span>

                  <span className={styles.statValue}>
                    {node.tg_tok_s} / {node.pp_tok_s}
                    <span className={styles.muted}>tok/s</span>
                  </span>
                </div>

                <div className={styles.statRow}>
                  <span className={styles.statLabel}>
                    <Network size={14} />
                    RTT
                  </span>

                  <span className={styles.statValue}>
                    {node.rtt_ms} ms
                  </span>
                </div>
              </div>

              <div
                className={styles.state}
                data-state={node.state}
              >
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