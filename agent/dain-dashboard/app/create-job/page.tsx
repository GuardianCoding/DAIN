"use client";

import { useState } from "react";
import styles from "./page.module.css";
import {
  Terminal,
  FileSearch,
  Search,
  Gauge,
  MessageSquare,
  Layers,
  Send,
} from "lucide-react";

// Mirrors contracts.py Job dataclass.
// POST /api/jobs { kind, payload, fanout, node_id }
type JobKind = "infer" | "exec" | "index" | "search" | "bench";

const KIND_OPTIONS: {
  id: JobKind;
  label: string;
  hint: string;
  icon: typeof MessageSquare;
}[] = [
  { id: "infer", label: "Infer", hint: "Run a prompt against the model", icon: MessageSquare },
  { id: "exec", label: "Exec", hint: "Sandboxed command on a node", icon: Terminal },
  { id: "index", label: "Index", hint: "Embed local files on a node", icon: FileSearch },
  { id: "search", label: "Search", hint: "Query the file index", icon: Search },
  { id: "bench", label: "Bench", hint: "Run llama-bench, record numbers", icon: Gauge },
];

const FANOUT_OPTIONS = [1, 2, 3, 4];

// Fake nodes for node targeting — replace with Abdallah's /api/nodes
const AVAILABLE_NODES = ["auto", "gpu-01", "office-01", "office-02", "nuc-01"];

type SubmitState = "idle" | "queued" | "error";

export default function CreateJob() {
  const [kind, setKind] = useState<JobKind>("infer");
  const [prompt, setPrompt] = useState("");
  const [fanout, setFanout] = useState(1);
  const [nodeId, setNodeId] = useState("auto");
  const [state, setState] = useState<SubmitState>("idle");

  const canFanout = kind === "infer" || kind === "bench";
  const charCount = prompt.length;
  const overCap = charCount > 8000; // ~2k tokens, §6.3 cap

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim() || overCap) return;

    setState("queued");
    try {
      // await fetch("/api/jobs", {
      //   method: "POST",
      //   headers: { "Content-Type": "application/json" },
      //   body: JSON.stringify({
      //     kind,
      //     payload: { prompt },
      //     fanout: canFanout ? fanout : 1,
      //     node_id: nodeId === "auto" ? null : nodeId,
      //   }),
      // });
    } catch {
      setState("error");
    }
  }

  return (
    <main className={styles.stage}>
      <header className={styles.header}>
        <div>
          <p className={styles.kicker}>D.A.I.N</p>
          <h1 className={styles.title}>Create Job</h1>
        </div>

        <span className={styles.badge}>
          <Layers size={13} />
          {canFanout ? `fanout ×${fanout}` : "single node"}
        </span>
      </header>

      <form className={styles.card} onSubmit={handleSubmit}>
        {/* Job kind */}
        <div className={styles.field}>
          <span className={styles.fieldLabel}>Kind</span>
          <div className={styles.kindGrid}>
            {KIND_OPTIONS.map(({ id, label, hint, icon: Icon }) => (
              <button
                type="button"
                key={id}
                className={styles.kindOption}
                data-active={kind === id}
                onClick={() => setKind(id)}
              >
                <Icon size={16} />
                <span className={styles.kindLabel}>{label}</span>
                <span className={styles.muted}>{hint}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Prompt / payload */}
        <div className={styles.field}>
          <div className={styles.fieldRow}>
            <span className={styles.fieldLabel}>Prompt</span>
            <span className={overCap ? styles.countWarn : styles.muted}>
              {charCount.toLocaleString()} / 8,000 chars
            </span>
          </div>
          <textarea
            className={styles.textarea}
            placeholder="Which node has the most free memory right now, and how much?"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={6}
          />
        </div>

        {/* Fanout */}
        {canFanout && (
          <div className={styles.field}>
            <span className={styles.fieldLabel}>Fanout</span>
            <div className={styles.pillRow}>
              {FANOUT_OPTIONS.map((n) => (
                <button
                  type="button"
                  key={n}
                  className={styles.pillOption}
                  data-active={fanout === n}
                  onClick={() => setFanout(n)}
                >
                  {n} node{n > 1 ? "s" : ""}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Target node — only meaningful outside fanout */}
        {!canFanout && (
          <div className={styles.field}>
            <span className={styles.fieldLabel}>Node</span>
            <div className={styles.pillRow}>
              {AVAILABLE_NODES.map((n) => (
                <button
                  type="button"
                  key={n}
                  className={styles.pillOption}
                  data-active={nodeId === n}
                  onClick={() => setNodeId(n)}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className={styles.footer}>

          <button
            type="submit"
            className={styles.submit}
            disabled={!prompt.trim() || overCap}
          >
            <Send size={15} />
            Run job
          </button>
        </div>
      </form>
    </main>
  );
}