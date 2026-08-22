const API_URL = process.env.REACT_APP_API_URL;

export async function getNodes() {
    const response = await fetch(`${API_URL}/nodes`)

    if (!response.ok) {
        throw new Error("Failed to get available nodes.")
    }

    return response.json();
}

export async function createJob(data) {
    const response = await fetch(`${API_URL}/jobs`, {
        method: "POST",
        headers: {
            "Content-Type" : "application/json",
        },
        body: JSON.stringify(data),
    });

    if (!response.ok) {
        throw new Error("Failed to create a new job.")
    }

    return response.json();
}