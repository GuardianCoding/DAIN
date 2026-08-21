from dataclasses import asdict

from fastapi import FastAPI

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
