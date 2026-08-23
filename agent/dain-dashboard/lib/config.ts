// lib/config.ts
// The single source of truth for where the control plane lives.
//
// Both values are NEXT_PUBLIC_* so Next inlines them into the client bundle at
// BUILD time — editing .env.local means restarting `next dev`, not just
// reloading the page.
//
// The defaults assume ctl is running on this machine. To point at the real
// cluster, set both in .env.local (see .env.example) to gpu-01's LAN address.
// Having defaults at all is deliberate: an unset NEXT_PUBLIC_FEED_URL made
// `new WebSocket(undefined)` throw synchronously, which killed the feed
// permanently and silently. A wrong-but-well-formed URL fails loudly and
// keeps retrying instead.

const DEFAULT_CTL_HOST = "127.0.0.1:8000";

/** Treat an unset OR blank env var as absent — a bare `NEXT_PUBLIC_API_URL=`
 *  in a .env file yields "", which `??` alone would happily accept. */
function resolve(value: string | undefined, fallback: string): string {
  const trimmed = value?.trim();
  return trimmed ? trimmed : fallback;
}

export const API_BASE = resolve(
  process.env.NEXT_PUBLIC_API_URL,
  `http://${DEFAULT_CTL_HOST}/api`,
);

export const FEED_URL = resolve(
  process.env.NEXT_PUBLIC_FEED_URL,
  `ws://${DEFAULT_CTL_HOST}/feed`,
);

// The agent runs in its own process (uvicorn agent.service:app --port 8100),
// separate from ctl for the same reason serve_head.py is: ctl must stay
// restartable without ending a conversation. So it gets its own address, and
// it is NOT under /api — that prefix belongs to the control plane.
const DEFAULT_AGENT_HOST = "127.0.0.1:8100";

export const AGENT_BASE = resolve(
  process.env.NEXT_PUBLIC_AGENT_URL,
  `http://${DEFAULT_AGENT_HOST}`,
);

// ctl serves REST under /api but mounts the socket at bare /feed, so the two
// URLs are not interchangeable. Dropping /api from the base is the easy
// mistake: the WebSocket still connects, every REST call 404s, and the UI
// half-works in a way that reads like a backend bug.
if (process.env.NODE_ENV !== "production" && !API_BASE.endsWith("/api")) {
  console.warn(
    `[dain] NEXT_PUBLIC_API_URL is "${API_BASE}" — it should end in /api, ` +
      `e.g. http://gpu-01.local:8000/api. Without it every REST call 404s ` +
      `while the feed keeps working.`,
  );
}
