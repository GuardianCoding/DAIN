"use client";

// app/create-job/page.tsx
//
// Every kind used to submit {prompt}, which only "infer" ever wanted. The node
// validates each kind's payload separately and 422s on anything else:
//
//   exec    payload.argv must be a list of strings   (+ cwd, timeout_s)
//   search  payload.query must not be empty          (+ limit, an int)
//   index   no payload — it re-scans the node's own files
//   infer   /infer does not exist on the node yet -> 404
//   bench   /bench does not exist on the node yet -> 404
//
// So the form is now kind-aware. infer and bench stay selectable on purpose:
// the moment those routes land they work with no frontend change, and until
// then the warning says exactly why the job will fail.

import { useState } from "react";
import styles from "./page.module.css";

import { createJob } from "../components/API/api";
import { useNodes } from "../../lib/feed/useNodes";
import { useJobs } from "../../lib/feed/useJobs";

import {
  Terminal, FileSearch, Search, Gauge, MessageSquare, Layers, Send,
  CheckCircle2, AlertTriangle,
} from "lucide-react";

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

// The node agent serves /health /profile /metrics /index /search /exec /infer.
// Only /bench is still mapped by the queue without being served, so it 404s.
const UNIMPLEMENTED_KINDS: JobKind[] = ["bench"];

const MAX_TOKEN_OPTIONS = [64, 128, 256, 512];
const DEFAULT_MAX_TOKENS = 256; // matches node/infer.py

const FANOUT_OPTIONS = [1, 2, 3, 4, 5];
const MAX_PROMPT_CHARS = 8000; // ~2k tokens, §6.3 cap
const DEFAULT_SEARCH_LIMIT = 5;
const SEARCH_LIMIT_OPTIONS = [3, 5, 10, 20];

type SubmitState = "idle" | "queued" | "success" | "error";

/** Split a command line into argv the way a shell would, minus expansion.
 *  Quote-aware because `echo "hello world"` must stay two arguments, not
 *  three — a plain .split(" ") silently corrupts any quoted argument. */
export function tokenizeCommand(input: string): string[] {
  const argv: string[] = [];
  let current = "";
  let quote: '"' | "'" | null = null;
  let hasCurrent = false;

  for (const char of input) {
    if (quote) {
      if (char === quote) quote = null;
      else current += char;
      continue;
    }

    if (char === '"' || char === "'") {
      quote = char;
      hasCurrent = true; // `""` is a real, empty argument
      continue;
    }

    if (/\s/.test(char)) {
      if (hasCurrent) {
        argv.push(current);
        current = "";
        hasCurrent = false;
      }
      continue;
    }

    current += char;
    hasCurrent = true;
  }

  if (hasCurrent) argv.push(current);
  return argv;
}

export default function CreateJob() {
  const nodes = useNodes();
  const { seedJob } = useJobs();

  const nodeNames = nodes.map((n) => n.id);

  const [kind, setKind] = useState<JobKind>("exec");
  const [prompt, setPrompt] = useState("");
  const [command, setCommand] = useState("");
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(DEFAULT_SEARCH_LIMIT);
  const [maxTokens, setMaxTokens] = useState(DEFAULT_MAX_TOKENS);
  const [fanout, setFanout] = useState(1);
  const [nodeId, setNodeId] = useState("auto");
  const [state, setState] = useState<SubmitState>("idle");
  const [errorDetail, setErrorDetail] = useState("");

  const canFanout = kind === "infer" || kind === "bench";
  const usesPrompt = kind === "infer" || kind === "bench";
  const isUnimplemented = UNIMPLEMENTED_KINDS.includes(kind);

  const argv = tokenizeCommand(command);
  const overCap = prompt.length > MAX_PROMPT_CHARS;

  function buildPayload(): object {
    switch (kind) {
      case "exec":
        return { argv };
      case "search":
        return { query: query.trim(), limit };
      case "index":
        return {};
      default:
        // infer and bench. max_tokens must be an int — node/infer.py rejects
        // a string with 422 rather than coercing it.
        return { prompt, max_tokens: maxTokens };
    }
  }

  /** Mirrors what the node will accept, so the button disables instead of the
   *  job coming back 422 from a machine we cannot see. */
  function isReady(): boolean {
    if (state === "queued") return false;
    if (kind === "exec") return argv.length > 0;
    if (kind === "search") return query.trim().length > 0;
    if (kind === "index") return true;
    return prompt.trim().length > 0 && !overCap;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!isReady()) return;

    setState("queued");
    setErrorDetail("");

    try {
      const job = await createJob(
        kind,
        buildPayload(),
        fanout,
        nodeId === "auto" ? null : nodeId,
      );
      seedJob(job); // row appears on /jobs immediately, before any feed frame

      setState("success");
      if (kind === "exec") setCommand("");
      if (kind === "search") setQuery("");
      if (usesPrompt) setPrompt("");
    } catch (err) {
      setState("error");
      setErrorDetail(err instanceof Error ? err.message : String(err));
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
          <span>Job queued. Track its progress on the jobs page.</span>
        </div>
      )}

      {state === "error" && (
        <div className={styles.errorBanner} role="alert">
          Couldn&apos;t queue the job.
          {errorDetail ? ` ${errorDetail}` : ""}
        </div>
      )}

      {isUnimplemented && (
        <div className={styles.errorBanner} role="alert">
          <AlertTriangle size={16} />
          <span>
            The node agent does not serve <code>/{kind}</code> yet, so this job
            will come back 404. Submit it if you want to see the failure path;
            it starts working the moment the route lands.
          </span>
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
                onClick={() => {
                  setKind(id);
                  setState("idle");
                }}
              >
                <Icon size={16} />
                <span className={styles.kindLabel}>{label}</span>
                <span className={styles.muted}>{hint}</span>
              </button>
            ))}
          </div>
        </div>

        {kind === "exec" && (
          <div className={styles.field}>
            <div className={styles.fieldRow}>
              <span className={styles.fieldLabel}>Command</span>
              <span className={styles.muted}>
                {argv.length} argument{argv.length === 1 ? "" : "s"}
              </span>
            </div>
            <input
              className={styles.textarea}
              placeholder="uname -a"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
            />
            <span className={styles.muted}>
              Sent as <code>payload.argv</code>. Quotes are respected; there is
              no shell, so pipes and globs are literal arguments.
              {argv.length > 0 && ` → ${JSON.stringify(argv)}`}
            </span>
          </div>
        )}

        {kind === "search" && (
          <>
            <div className={styles.field}>
              <span className={styles.fieldLabel}>Query</span>
              <input
                className={styles.textarea}
                placeholder="where is the tensor split computed?"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <span className={styles.muted}>
                Needs an index on the node first — Search returns 409 until
                Index has run.
              </span>
            </div>
            <div className={styles.field}>
              <span className={styles.fieldLabel}>Limit</span>
              <div className={styles.pillRow}>
                {SEARCH_LIMIT_OPTIONS.map((n) => (
                  <button
                    type="button"
                    key={n}
                    className={styles.pillOption}
                    data-active={limit === n}
                    onClick={() => setLimit(n)}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
          </>
        )}

        {kind === "index" && (
          <div className={styles.field}>
            <span className={styles.fieldLabel}>Index</span>
            <span className={styles.muted}>
              Takes no payload — the node re-scans and embeds its own configured
              paths. Returns 503 if the embedding model is not cached locally;
              see scripts/fetch_embed_model.py.
            </span>
          </div>
        )}

        {usesPrompt && (
          <div className={styles.field}>
            <div className={styles.fieldRow}>
              <span className={styles.fieldLabel}>Prompt</span>
              <span className={overCap ? styles.countWarn : styles.muted}>
                {prompt.length.toLocaleString()} /{" "}
                {MAX_PROMPT_CHARS.toLocaleString()} chars
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
        )}

        {usesPrompt && (
          <div className={styles.field}>
            <span className={styles.fieldLabel}>Max tokens</span>
            <div className={styles.pillRow}>
              {MAX_TOKEN_OPTIONS.map((n) => (
                <button
                  type="button"
                  key={n}
                  className={styles.pillOption}
                  data-active={maxTokens === n}
                  onClick={() => setMaxTokens(n)}
                >
                  {n}
                </button>
              ))}
            </div>
            <span className={styles.muted}>
              Longer generations make the fan-out speed difference easier to
              see, but every node has to finish before the job completes.
            </span>
          </div>
        )}

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
            {nodeNames.length === 0 && (
              <span className={styles.countWarn}>waiting for nodes to join…</span>
            )}
          </div>
        )}

        <div className={styles.footer}>
          <button type="submit" className={styles.submit} disabled={!isReady()}>
            <Send size={15} />
            {state === "queued" ? "Running…" : "Run job"}
          </button>
        </div>
      </form>
    </main>
  );
}
