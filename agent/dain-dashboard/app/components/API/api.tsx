const API_URL = process.env.NEXT_PUBLIC_API_URL;

export async function getNodes() {
    const response = await fetch(`${API_URL}/nodes`);

    if (!response.ok) {
        throw new Error("Failed to get available nodes.");
    }

    return response.json();
}

export async function createJob(
    kind: string,
    payload: object,
    fanout: number,
    node_id: string | null
) {
    const response = await fetch(`${API_URL}/jobs`, {
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
        throw new Error("Failed to create job.");
    }

    return response.json();
}