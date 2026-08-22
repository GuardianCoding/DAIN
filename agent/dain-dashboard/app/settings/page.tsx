"use client";

// app/settings/page.tsx
//
// The sidebar has always linked here and there was never a route, so it 404'd.
// Rather than delete the link, this is the page that would have saved the most
// time: it shows the endpoints the bundle was actually built with.
//
// That matters because NEXT_PUBLIC_* is inlined at BUILD time. Editing
// .env.local and reloading the page changes nothing — you have to restart
// `next dev`. Read-only on purpose: these values cannot change at runtime.

import styles from "./page.module.css";
import { API_BASE, FEED_URL } from "../../lib/config";
import { useFeed } from "../../lib/feed/FeedProvider";
import { Check, X, AlertTriangle } from "lucide-react";

export default function SettingsPage() {
  const { connected, nodes, jobs } = useFeed();

  // ctl serves REST under /api but mounts the socket at bare /feed. Dropping
  // /api leaves the feed working while every REST call 404s — a half-broken
  // state that reads like a backend fault.
  const apiBaseLooksWrong = !API_BASE.endsWith("/api");

  return (
    <main className={styles.stage}>
      <header className={styles.header}>
        <div>
          <p className={styles.kicker}>D.A.I.N</p>
          <h1 className={styles.title}>Settings</h1>
        </div>
        <span className={styles.status} data-connected={connected}>
          {connected ? <Check size={13} /> : <X size={13} />}
          {connected ? "connected" : "disconnected"}
        </span>
      </header>

      <section className={styles.card}>
        <h2 className={styles.sectionTitle}>Control plane</h2>

        <dl className={styles.rows}>
          <div className={styles.row}>
            <dt>API base</dt>
            <dd className={styles.mono}>{API_BASE}</dd>
          </div>
          <div className={styles.row}>
            <dt>Feed socket</dt>
            <dd className={styles.mono}>{FEED_URL}</dd>
          </div>
          <div className={styles.row}>
            <dt>Feed state</dt>
            <dd>{connected ? "open" : "closed — retrying every 2s"}</dd>
          </div>
          <div className={styles.row}>
            <dt>Nodes known</dt>
            <dd>{nodes.length}</dd>
          </div>
          <div className={styles.row}>
            <dt>Jobs tracked</dt>
            <dd>{jobs.length}</dd>
          </div>
        </dl>

        {apiBaseLooksWrong && (
          <p className={styles.warn}>
            <AlertTriangle size={14} />
            <span>
              The API base should end in <code>/api</code>. As set, REST calls
              will 404 while the feed keeps working.
            </span>
          </p>
        )}

        <p className={styles.note}>
          Set in <code>agent/dain-dashboard/.env.local</code> as{" "}
          <code>NEXT_PUBLIC_API_URL</code> and <code>NEXT_PUBLIC_FEED_URL</code>
          . These are inlined at build time — <strong>restart the dev server
          after editing;</strong> reloading the page is not enough.
        </p>
      </section>

      <section className={styles.card}>
        <h2 className={styles.sectionTitle}>Known gaps</h2>
        <ul className={styles.gaps}>
          <li>
            <code>infer</code> and <code>bench</code> jobs 404 — the node agent
            serves neither route yet.
          </li>
          <li>
            <code>index</code> returns 503 unless the embedding model is cached
            on the node. See <code>scripts/fetch_embed_model.py</code>.
          </li>
          <li>
            <code>search</code> returns 409 until <code>index</code> has run on
            that node.
          </li>
          <li>
            <code>GET /api/plan</code> still returns the mock assignment; the
            real scheduler needs calibrated <code>tg_tok_s</code> per node.
          </li>
        </ul>
      </section>
    </main>
  );
}
