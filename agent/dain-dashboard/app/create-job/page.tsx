"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import styles from "./page.module.css";
import { getNodes, createJob } from "../components/API/api";
import {
  Terminal, FileSearch, Search, Gauge, MessageSquare, Layers, Send, CheckCircle2,
} from "lucide-react";

type JobKind = "infer" | "exec" | "index" | "search" | "bench";

const KIND_OPTIONS: { id: JobKind; label: string; hint: string; icon: typeof MessageSquare }[] = [
  { id: "infer", label: "Infer", hint: "Run a prompt against the model", icon: MessageSquare },
  { id: "exec", label: "Exec", hint: "Sandboxed command on a node", icon: Terminal },
  { id: "index", label: "Index", hint: "Embed local files on a node", icon: FileSearch },
  { id: "search", label: "Search", hint: "Query the file index", icon: Search },
  { id: "bench", label: "Bench", hint: "Run llama-bench, record numbers", icon: Gauge },
];

const FANOUT_OPTIONS = [1, 2, 3, 4, 5];
type SubmitState = "idle" | "queued" | "success" | "error";

export default function CreateJob() {
  const [nodes, setNodes] = useState<[]>([]);
  const [nodesError, setNodesError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getNodes()
      .then((res) => { if (!cancelled) setNodes(res); })
      .catch((err) => { if (!cancelled) setNodesError(String(err)); });
    return () => { cancelled = true; };
  }, []);

  const nodeNames = nodes.map((n) => n.id);

  const [kind, setKind] = useState("infer");
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
      await createJob(
        kind,
        { prompt },
        fanout,
        nodeId === "auto" ? null : nodeId
      );

      setState("success");
      setPrompt("");
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

      {state === "success" && (
        <div className={styles.successBanner} role="status">
          <CheckCircle2 size={16} />
          <span>
            Job queued. Track its progress on the jobs page.
          </span>
        </div>
      )}

      {state === "error" && (
        <div className={styles.errorBanner} role="alert">
          Couldn't queue the job. Check the fanout settings and try again.
        </div>
      )}

      <form className={styles.card} onSubmit={handleSubmit}>
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

        {!canFanout && (
          <div className={styles.field}>
            <span className={styles.fieldLabel}>Node</span>
            <div className={styles.pillRow}>
              <button
                type="button"
                className={styles.pillOption}
                data-active={nodeId === "auto"}
                onClick={() => setNodeId("auto")}
              >
                auto
              </button>
              {nodeNames.map((n) => (
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
            {nodesError && <span className={styles.countWarn}>nodes unavailable: {nodesError}</span>}
          </div>
        )}

        <div className={styles.footer}>
          <button
            type="submit"
            className={styles.submit}
            disabled={!prompt.trim() || overCap || state === "queued"}
          >
            <Send size={15} />
            {state === "queued" ? "Running…" : "Run job"}
          </button>
        </div>
      </form>
    </main>
  );
}