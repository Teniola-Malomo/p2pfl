import argparse
import time
import threading
import socket
from datetime import datetime
from queue import Queue, Empty
import logging
import grpc
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel
from p2pfl.node import Node
from p2pfl.management.logger import logger
from p2pfl.utils.utils import set_test_settings

# -------------------- Initialization -------------------- #
set_test_settings()

log_stream_queue = Queue()

class QueueStreamHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        log_stream_queue.put(msg)

stream_handler = QueueStreamHandler()
stream_handler.setFormatter(logging.Formatter('[%(asctime)s | %(node)s | %(levelname)s ]: %(message)s', "%Y-%m-%dT%H:%M:%S"))
logger.add_handler(stream_handler)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

node = None
created_at = datetime.utcnow().isoformat()

IP_NODE_MAP = {
    "172.20.0.2": {"node_id": "node1", "name": "GCSNode"},
    "172.20.0.3": {"node_id": "node2", "name": "GCPNode"},
    "172.20.0.4": {"node_id": "node3", "name": "SMSNode"},
}

ip = socket.gethostbyname(socket.gethostname())
node_meta = IP_NODE_MAP.get(ip, {"node_id": "unknown", "name": "UnnamedNode"})

# -------------------- Training Lifecycle -------------------- #
def launch_node(host, port, wait_peers, connect_list):
    global node
    address = f"{host}:{port}"
    logger.info(address, f"Node starting at {address}")

    node = Node(
        LightningModel(MLP()),
        P2PFLDataset.from_huggingface("p2pfl/MNIST"),
        address=address,
        learner=LightningLearner
    )

    node.start()
    logger.experiment_started(address, node.state.exp_name)

    if connect_list:
        for peer in connect_list:
            logger.info(address, f"Connecting to peer {peer}")
            node.connect(peer)
        time.sleep(4)

    if wait_peers > 0:
        logger.info(address, f"Waiting for {wait_peers} peers...")
        while len(node.get_neighbors()) < wait_peers:
            logger.info(address, f"Known peers so far: {list(node.get_neighbors().keys())}")
            time.sleep(5)

    logger.info(address, "Starting federated learning")
    node.set_start_learning(rounds=2, epochs=1)

    while node.state.round is not None:
        time.sleep(1)

    logger.experiment_finished(address)
    logger.info(address, "Node finished. Stopping.")
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
    return {"neighbors": list(neighbors.keys())}

@app.get("/communication")
def communication_summary():
    neighbors = node.get_neighbors()
    detailed = {}

    state_mapping = {
        grpc.ChannelConnectivity.IDLE: "IDLE",
        grpc.ChannelConnectivity.CONNECTING: "CONNECTING",
        grpc.ChannelConnectivity.READY: "READY",
        grpc.ChannelConnectivity.TRANSIENT_FAILURE: "TRANSIENT_FAILURE",
        grpc.ChannelConnectivity.SHUTDOWN: "SHUTDOWN",
    }

    for addr, (channel, stub, timestamp) in neighbors.items():
        try:
            module = stub.__class__.__module__
            connection_type = "gRPC" if "grpc" in module else module.split('.')[0]

            if hasattr(channel, "get_state"):
                grpc_state = channel.get_state(True)
                channel_state = state_mapping.get(
                    grpc_state,
                    grpc_state.name if hasattr(grpc_state, "name") else str(grpc_state)
                )
            else:
                channel_state = "CONNECTING"

            detailed[addr] = {
                "connection_type": connection_type,
                "channel": type(channel).__name__,
                "stub": type(stub).__name__,
                "last_contact": timestamp,
                "last_contact_human": datetime.fromtimestamp(timestamp).isoformat(),
                "channel_state": channel_state
            }

        except Exception as e:
            detailed[addr] = {
                "error": str(e),
                "connection_type": "Unknown",
                "channel_state": "CONNECTING"
            }

    return {"communications": detailed}

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
    parser.add_argument("--wait_peers", type=int, default=0)
    parser.add_argument("--connect", type=str, nargs="*", help="List of nodes to connect to (ip:port)")
    parser.add_argument("--api_port", type=int, default=8000)
    args = parser.parse_args()

    threading.Thread(
        target=launch_node,
        args=(args.host, args.port, args.wait_peers, args.connect),
        daemon=True
    ).start()

    uvicorn.run(app, host="0.0.0.0", port=args.api_port)
