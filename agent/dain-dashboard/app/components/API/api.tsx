// One-off REST calls against the control plane. Anything that needs to stay
// live belongs on the /feed socket instead — see lib/feed/FeedProvider.tsx.

import { API_BASE } from "../../../lib/config";

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
