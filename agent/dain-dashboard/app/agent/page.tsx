"use client";

// app/agent/page.tsx
//
// Talk to the pool. The agent runs in its own process (agent.service on :8100),
// NOT inside ctl and not inside Next — see the docstring in agent/service.py.
//
// Two things this page does that a plain chat box would not:
//
//   1. It shows the tool calls. The whole claim is that answers come from live
//      cluster state rather than the model's imagination, and an answer with no
//      visible work behind it is indistinguishable from a guess. Every call is
//      expandable down to the raw text the tool returned.
//   2. It keeps the transcript. agent.service is stateless by design, so the
//      conversation lives here and goes back as `history` on the next turn.
//      Restarting the service mid-demo costs nothing.
//
// Tool calls also become ctl jobs, so anything that happens here shows up on
// /feed at the same time — open /dashboard alongside and the nodes light up.

import { useCallback, useEffect, useRef, useState } from "react";
import styles from "./page.module.css";

import {
  askAgent,
  getAgentHealth,
  type AgentToolCall,
} from "../components/API/api";
import { AGENT_BASE } from "../../lib/config";
import { Send, RotateCcw, Loader2 } from "lucide-react";

type Turn = {
  prompt: string;
  text: string;
  toolCalls: AgentToolCall[];
  hitTurnCap: boolean;
  failed: boolean;
};

type Health = { endpoint: string; ctl: string } | null;

const SUGGESTIONS = [
  "Which machine has the most free memory right now?",
  "How would gpt-oss-20b be split across the pool?",
  "What is every node doing at the moment?",
];

export default function AgentPage() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [history, setHistory] = useState<unknown[]>([]);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<Health>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // One probe on mount, purely so the header can name where this thinks and
    // where it acts. A failure here is not fatal — the composer still works and
    // the real error, if any, arrives with the first message.
    let cancelled = false;

    getAgentHealth()
      .then((body) => {
        if (cancelled) return;
        setHealth({ endpoint: body.endpoint, ctl: body.ctl });
        setReachable(true);
      })
      .catch(() => {
        if (!cancelled) setReachable(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, busy]);

  const send = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || busy) return;

      setPrompt("");
      setBusy(true);

      try {
        const reply = await askAgent(question, history);
        setTurns((prev) => [
          ...prev,
          {
            prompt: question,
            text: reply.text,
            toolCalls: reply.tool_calls,
            hitTurnCap: reply.hit_turn_cap,
            failed: false,
          },
        ]);
        setHistory(reply.messages);
        setReachable(true);
      } catch (error) {
        // A failed turn is shown in place and the history is left untouched, so
        // the next message continues from the last good state rather than
        // carrying a broken exchange the model would have to reason around.
        setTurns((prev) => [
          ...prev,
          {
            prompt: question,
            text: error instanceof Error ? error.message : String(error),
            toolCalls: [],
            hitTurnCap: false,
            failed: true,
          },
        ]);
        setReachable(false);
      } finally {
        setBusy(false);
      }
    },
    [busy, history],
  );

  function reset() {
    setTurns([]);
    setHistory([]);
  }

  return (
    <main className={styles.stage}>
      <header className={styles.header}>
        <div>
          <p className={styles.kicker}>Agent</p>
          <h1 className={styles.title}>Ask the pool</h1>
        </div>
        <div className={styles.endpoints}>
          <div className={styles.status}>
            <span
              className={styles.dot}
              data-state={
                reachable === null ? "unknown" : reachable ? "up" : "down"
              }
            />
            {reachable === false ? "agent unreachable" : AGENT_BASE}
          </div>
          {health && (
            <>
              <div>
                thinks on <b>{health.endpoint}</b>
              </div>
              <div>
                acts through <b>{health.ctl}</b>
              </div>
            </>
          )}
        </div>
      </header>

      <section className={styles.transcript}>
        {turns.length === 0 && !busy && (
          <div className={styles.empty}>
            <h2>Its tools are the cluster.</h2>
            Ask about free memory, placement, files on any node, or run a
            sandboxed command. Answers come from the live registry, not from
            what the model remembers — every tool call it makes is shown below
            the answer, and each one also appears on the dashboard feed.
            <div className={styles.suggestions}>
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  className={styles.suggestion}
                  onClick={() => send(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn, index) => (
          <div key={index} className={styles.turn}>
            <div className={`${styles.bubble} ${styles.user}`}>
              {turn.prompt}
            </div>

            {turn.toolCalls.length > 0 && (
              <div className={styles.work}>
                <div className={styles.workHeader}>
                  {turn.toolCalls.length} tool call
                  {turn.toolCalls.length === 1 ? "" : "s"}
                </div>
                {turn.toolCalls.map((call, callIndex) => (
                  <details key={callIndex} className={styles.call}>
                    <summary className={styles.callSummary}>
                      <span className={styles.callName}>{call.name}</span>
                      <span className={styles.callArgs}>
                        {Object.keys(call.arguments ?? {}).length > 0
                          ? JSON.stringify(call.arguments)
                          : "()"}
                      </span>
                    </summary>
                    <pre className={styles.callResult}>{call.result}</pre>
                  </details>
                ))}
              </div>
            )}

            <div
              className={`${styles.bubble} ${
                turn.failed ? styles.failed : styles.assistant
              }`}
            >
              {turn.text}
            </div>

            {turn.hitTurnCap && (
              <div className={styles.capped}>
                Stopped at the turn cap rather than guessing.
              </div>
            )}
          </div>
        ))}

        {busy && (
          <div className={styles.thinking}>
            <Loader2 size={13} /> thinking, and calling the cluster…
          </div>
        )}
        <div ref={endRef} />
      </section>

      <form
        className={styles.composer}
        onSubmit={(event) => {
          event.preventDefault();
          send(prompt);
        }}
      >
        <textarea
          className={styles.input}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Ask about the cluster…"
          rows={1}
          disabled={busy}
          onKeyDown={(event) => {
            // Enter sends, Shift+Enter breaks the line. A multi-line prompt is
            // rare here and needing the mouse for every question is not.
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              send(prompt);
            }
          }}
        />
        <button
          type="button"
          className={styles.reset}
          onClick={reset}
          disabled={busy || turns.length === 0}
          title="Start a new conversation"
        >
          <RotateCcw size={16} />
        </button>
        <button
          type="submit"
          className={styles.send}
          disabled={busy || !prompt.trim()}
        >
          <Send size={16} />
          Ask
        </button>
      </form>
    </main>
  );
}
