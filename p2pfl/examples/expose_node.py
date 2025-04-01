from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import socket
import threading
import time

from p2pfl.communication.protocols.neighbors import Neighbors
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel
from p2pfl.node import Node
from p2pfl.utils.utils import set_test_settings

# ----- FastAPI App Setup -----
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Node Metadata -----
IP_NODE_MAP = {
    "172.20.0.2": {"node_id": "node1", "name": "GCSNode"},
    "172.20.0.3": {"node_id": "node2", "name": "GCPNode"},
    "172.20.0.4": {"node_id": "node3", "name": "SMSNode"},
}

ip = socket.gethostbyname(socket.gethostname())
node_meta = IP_NODE_MAP.get(ip, {"node_id": "unknown", "name": "UnnamedNode"})
created_at = datetime.utcnow().isoformat()

# ----- Global State -----
set_test_settings()
node = None
neighbors = Neighbors(self_addr=node_meta["node_id"])


# ----- Endpoints -----

@app.get("/info")
def get_node_info():
    return {
        "node_id": node_meta["node_id"],
        "name": node_meta["name"],
        "ip": ip,
        "status": "running" if node else "not started",
        "created_at": created_at,
        "last_seen": datetime.utcnow().isoformat(),
        "neighbors": list(node.get_neighbors().keys()) if node else [],
        "round": node.state.round if node and node.state else None
    }

@app.get("/neighbors")
def get_neighbors():
    print()
    print(node)
    print()
    if node:
        return node.get_neighbors()
    return []

@app.post("/start-node")
def start_node(wait_peers: int = 1):
    def run_node():
        global node
        address = f"{ip}:5001"  # This is for the internal gRPC comms
        print(f"🚀 Starting node at {address}")

        node_instance = Node(
            LightningModel(MLP()),
            P2PFLDataset.from_huggingface("p2pfl/MNIST"),
            address=address,
            learner=LightningLearner
        )
        node_instance.start()
        globals()["node"] = node_instance  # Save globally so /info can access

        print(f"⏳ Waiting for {wait_peers} peers...")
        while len(node_instance.get_neighbors().keys()) < wait_peers:
            print(f"🔗 Connected peers: {node_instance.get_neighbors().keys()}")
            time.sleep(7)

        print("✅ Enough peers. Starting federated learning.")
        node_instance.set_start_learning(rounds=2, epochs=1)

        while True:
            time.sleep(1)
            if node_instance.state.round is None:
                break

        print("🎉 Node finished training. Stopping.")
        node_instance.stop()

    # Start in background thread
    threading.Thread(target=run_node, daemon=True).start()
    return {"message": f"Node is starting on {ip}:5001"}
