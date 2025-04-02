# from fastapi import FastAPI
# from p2pfl.communication.protocols.neighbors import Neighbors

# app = FastAPI()

# # TEMP hardcoded — this gets replaced at runtime by real Neighbors
# neighbors = Neighbors(self_addr="node1")

# @app.get("/neighbors")
# def get_neighbors():
#     return neighbors.get_all()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import socket

app = FastAPI()

# Allow frontend on localhost:5173 to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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



# from p2pfl.management.logger import logger

# @app.get("/metrics")
# def get_all_metrics():
#     return {
#         "local": logger.get_local_logs(),
#         "global": logger.get_global_logs()
#     }

# @app.get("/metrics/{experiment}")
# def get_experiment_metrics(experiment: str):
#     return {
#         "local": logger.local_metrics.get_experiment_logs(experiment),
#         "global": logger.global_metrics.get_experiment_logs(experiment)
#     }

# @app.get("/metrics/{experiment}/round/{round}")
# def get_experiment_round_metrics(experiment: str, round: int):
#     return logger.local_metrics.get_experiment_round_logs(experiment, round)

# @app.get("/metrics/{experiment}/round/{round}/node/{node}")
# def get_experiment_round_node_metrics(experiment: str, round: int, node: str):
#     return logger.local_metrics.get_experiment_round_node_logs(experiment, round, node)
