# from fastapi import FastAPI
# from p2pfl.communication.protocols.neighbors import Neighbors

# app = FastAPI()

# # TEMP hardcoded — this gets replaced at runtime by real Neighbors
# neighbors = Neighbors(self_addr="node1")

# @app.get("/neighbors")
# def get_neighbors():
#     return neighbors.get_all()


from fastapi import FastAPI
from datetime import datetime
import socket

app = FastAPI()

# Maps IP addresses to logical IDs and names
IP_NODE_MAP = {
    "172.20.0.2": {"node_id": "node1", "name": "GCSNode"},
    "172.20.0.3": {"node_id": "node2", "name": "GCPNode"},
    "172.20.0.4": {"node_id": "node3", "name": "SMSNode"},
}

# Get own IP address inside container
ip = socket.gethostbyname(socket.gethostname())

# Look up metadata for this node
node_meta = IP_NODE_MAP.get(ip, {"node_id": "unknown", "name": "UnnamedNode"})

# Track when this container was created
created_at = datetime.utcnow().isoformat()

@app.get("/info")
def get_node_info():
    return {
        "node_id": node_meta["node_id"],
        "name": node_meta["name"],
        "ip": ip,
        "status": "running",  # Will improve later
        "created_at": created_at,
        "last_seen": datetime.utcnow().isoformat(),
    }

