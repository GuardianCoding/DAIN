from dataclasses import asdict

from fastapi import FastAPI, WebSocket

from ctl.mock import MOCK_NODES

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
    response_payload = {"type": "topology", "nodes": get_nodes()}
    await websocket.send_json(response_payload)
