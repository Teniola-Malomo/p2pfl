import argparse
import time
import threading
import socket
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import uvicorn

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel
from p2pfl.node import Node
from p2pfl.management.logger import logger
from p2pfl.utils.utils import set_test_settings

set_test_settings()

# ------------------------------
# FastAPI SETUP
# ------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Globals
# ------------------------------
node = None
created_at = datetime.utcnow().isoformat()

# For human reference
IP_NODE_MAP = {
    "172.20.0.2": {"node_id": "node1", "name": "GCSNode"},
    "172.20.0.3": {"node_id": "node2", "name": "GCPNode"},
    "172.20.0.4": {"node_id": "node3", "name": "SMSNode"},
}
ip = socket.gethostbyname(socket.gethostname())
node_meta = IP_NODE_MAP.get(ip, {"node_id": "unknown", "name": "UnnamedNode"})


# ------------------------------
# Training logic
# ------------------------------
def start_node(host, port, wait_peers):
    global node

    address = f"{host}:{port}"
    print(f"Node starting at {address}")

    node = Node(
        LightningModel(MLP()),
        P2PFLDataset.from_huggingface("p2pfl/MNIST"),
        address=address,
        learner=LightningLearner
    )

    node.start()

    logger.experiment_started(address, node.state.exp_name)

    print(f"Waiting for {wait_peers} peers...")
    while len(node.get_neighbors().keys()) < wait_peers:
        print(f"🔗 Known peers so far: {node.get_neighbors().keys()}")
        time.sleep(5)

    print("Enough peers connected. Starting federated learning")
    node.set_start_learning(rounds=2, epochs=1)

    while True:
        time.sleep(1)
        if node.state.round is None:
            break

    logger.experiment_finished(address, "exp_1")
    print("Node finished. Stopping.")
    node.stop()


# ------------------------------
# FastAPI Endpoints
# ------------------------------
@app.get("/")
def root():
    return {"message": "Node REST API is alive!"}

@app.get("/info")
def get_node_info():
    if node is None:
        return {"error": "Node not initialized"}
    return {
        "node_id": node_meta["node_id"],
        "name": node_meta["name"],
        "ip": ip,
        "status": node.state.status,
        "created_at": created_at,
        "last_seen": datetime.utcnow().isoformat(),
    }

# @app.get("/neighbors")
# def get_neighbors():
#     neighbors = node.get_neighbors()
#     return {"neighbors": list(neighbors.keys())}

@app.get("/neighbors")
def get_neighbors():
    neighbors = node.get_neighbors()
    simplified = {addr: str(type(channel)) for addr, channel in neighbors.items()}
    return {"neighbors": simplified}


@app.get("/training-status")
def get_training_status():
    if node is None:
        return {"error": "Node not initialized"}
    return {
        "status": node.state.status,
        "experiment": node.state.exp_name,
        "round": node.state.round,
        "total_rounds": node.state.total_rounds,
        "simulation": node.state.simulation,
    }

@app.get("/metrics")
def get_metrics():
    return {
        "local": logger.get_local_logs(),
        "global": logger.get_global_logs()
    }


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--wait_peers", type=int, default=1)
    parser.add_argument("--api_port", type=int, default=8000)
    args = parser.parse_args()

    # Start node in a background thread
    threading.Thread(
        target=start_node,
        args=(args.host, args.port, args.wait_peers),
        daemon=True
    ).start()

    # Start REST API
    uvicorn.run(app, host="0.0.0.0", port=args.api_port)
