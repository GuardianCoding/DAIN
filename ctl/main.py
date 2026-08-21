from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ctl.mock import MOCK_NODES

import asyncio

import time

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
            counter += 1
            payload = {
              "type": "metrics",
              "timestamp": time.time(),
              "nodes": [
                {
                  "id": "gpu-01",
                  "cpu_percent": 25 + counter % 10,
                  "ram_used_mb": 18432,
                  "gpu_percent": 40 + counter % 10 
                }
              ]
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("disconnected")
