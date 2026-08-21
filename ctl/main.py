import asyncio
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ctl.mock import MOCK_NODES, make_mock_metrics

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def check_health():
    return {"status": "ok"}


@app.get("/api/nodes")
def get_nodes():
    return [asdict(node) for node in MOCK_NODES]


@app.websocket("/feed")
async def send_feed(websocket: WebSocket):
    await websocket.accept()
    counter = 0
    response_payload = {"type": "topology", "nodes": get_nodes()}
    await websocket.send_json(response_payload)
    try:
        while True:
            metrics = make_mock_metrics(counter)
            payload = {
                "type": "metrics",
                "nodes": [asdict(metric) for metric in metrics],
            }
            await websocket.send_json(payload)
            counter += 1
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("disconnected")
