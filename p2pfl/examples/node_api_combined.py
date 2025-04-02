import argparse
import time
import threading
import socket
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from queue import Queue, Empty
import uvicorn
import logging

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel
from p2pfl.node import Node
from p2pfl.management.logger import logger
from p2pfl.utils.utils import set_test_settings

set_test_settings()

# -------------------- Shared Log Queue -------------------- #
log_stream_queue = Queue()

class QueueStreamHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        log_stream_queue.put(msg)

# Attach to the p2pfl logger instance
stream_handler = QueueStreamHandler()
stream_handler.setFormatter(logging.Formatter('[%(asctime)s | %(node)s | %(levelname)s ]: %(message)s', "%Y-%m-%dT%H:%M:%S"))
logger.add_handler(stream_handler)

# -------------------- FastAPI Setup -------------------- #
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Globals -------------------- #
node = None
created_at = datetime.utcnow().isoformat()
IP_NODE_MAP = {
    "172.20.0.2": {"node_id": "node1", "name": "GCSNode"},
    "172.20.0.3": {"node_id": "node2", "name": "GCPNode"},
    "172.20.0.4": {"node_id": "node3", "name": "SMSNode"},
}
ip = socket.gethostbyname(socket.gethostname())
node_meta = IP_NODE_MAP.get(ip, {"node_id": "unknown", "name": "UnnamedNode"})

# -------------------- Training logic -------------------- #
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
    node.set_start_learning(rounds=1, epochs=1)

    while True:
        time.sleep(1)
        if node.state.round is None:
            break

    logger.experiment_finished(address)
    print("Node finished. Stopping.")
    node.stop()

# -------------------- API Endpoints -------------------- #
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

@app.get("/neighbors")
def get_neighbors():
    neighbors = node.get_neighbors()
    detailed = {}

    for addr, (channel, stub, timestamp) in neighbors.items():
        try:
            # This gives some meaningful internal state of the gRPC channel
            channel_state = channel.get_state(True) if hasattr(channel, "get_state") else "Unknown"
            detailed[addr] = {
                "channel_type": str(type(channel)),
                "stub_type": str(type(stub)),
                "last_contact": timestamp,
                "channel_state": str(channel_state),
            }
        except Exception as e:
            detailed[addr] = {"error": str(e)}

    return {"neighbors": detailed}

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

@app.get("/stream-logs")
def stream_logs():
    def event_stream():
        while True:
            try:
                log = log_stream_queue.get(timeout=1)
                yield f"data: {log}\n\n"
            except Empty:
                continue

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# -------------------- Entry Point -------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--wait_peers", type=int, default=1)
    parser.add_argument("--api_port", type=int, default=8000)
    args = parser.parse_args()

    threading.Thread(
        target=start_node,
        args=(args.host, args.port, args.wait_peers),
        daemon=True
    ).start()

    uvicorn.run(app, host="0.0.0.0", port=args.api_port)
