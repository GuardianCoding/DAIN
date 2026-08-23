// One-off REST calls against the control plane. Anything that needs to stay
// live belongs on the /feed socket instead — see lib/feed/FeedProvider.tsx.

import { AGENT_BASE, API_BASE } from "../../../lib/config";

/** Snapshot of the registry. Prefer useNodes() in components: it reads the
 *  same data from the feed, updates itself, and survives ctl being down. */
export async function getNodes() {
    const response = await fetch(`${API_BASE}/nodes`);

    if (!response.ok) {
        throw new Error(
            `Failed to get available nodes: ${response.status} ${response.statusText}`
        );
    }

    return response.json();
}

export async function createJob(
    kind: string,
    payload: object,
    fanout: number,
    node_id: string | null
) {
    const response = await fetch(`${API_BASE}/jobs`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            kind,
            payload,
            fanout,
            node_id,
        }),
    });

    if (!response.ok) {
        // ctl returns 422 for a bad kind/fanout and 503 when no node can take
        // the job; surfacing the body makes the difference visible in the UI.
        const detail = await response.text().catch(() => "");
        throw new Error(
            `Failed to create job: ${response.status} ${response.statusText}${
                detail ? ` — ${detail}` : ""
            }`
        );
    }

    return response.json();
}

// --- the agent -------------------------------------------------------------
// A different process from ctl (agent.service on :8100), so a different base.
// Its tool calls still become ctl jobs, which is why they show up on /feed and
// the dashboard draws them while a conversation is running.

export type AgentToolCall = {
    name: string;
    arguments: Record<string, unknown>;
    result: string;
};

export type AgentReply = {
    text: string;
    tool_calls: AgentToolCall[];
    turns: number;
    hit_turn_cap: boolean;
    /** The whole conversation. Send it straight back as `history` next turn —
     *  the service is stateless, so this is where the thread actually lives. */
    messages: unknown[];
};

/** ctl-style {"detail": ...} if there is one, else something readable. */
async function errorDetail(response: Response, fallback: string): Promise<string> {
    try {
        const body = await response.json();
        if (body && typeof body.detail === "string") return body.detail;
    } catch {
        // non-JSON body; fall through
    }
    return fallback;
}

export async function askAgent(
    prompt: string,
    history: unknown[] = []
): Promise<AgentReply> {
    let response: Response;

    try {
        response = await fetch(`${AGENT_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, history }),
        });
    } catch (cause) {
        // Distinct from a 503: nothing is listening at all, which usually means
        // the agent service was never started rather than the head being down.
        throw new Error(
            `No agent service at ${AGENT_BASE}. Start it with ` +
                `uv run uvicorn agent.service:app --port 8100 ` +
                `(${cause instanceof Error ? cause.message : String(cause)})`
        );
    }

    if (!response.ok) {
        // 503 means the head is unreachable or erroring, and that message names
        // the endpoint and the command that starts it — worth showing verbatim.
        throw new Error(
            await errorDetail(
                response,
                `Agent request failed: ${response.status} ${response.statusText}`
            )
        );
    }

    return response.json();
}

export async function getAgentHealth() {
    const response = await fetch(`${AGENT_BASE}/health`);

    if (!response.ok) {
        throw new Error(`Agent health check failed: ${response.status}`);
    }

    return response.json();
}
