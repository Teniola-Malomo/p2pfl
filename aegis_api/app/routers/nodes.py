from fastapi import APIRouter
# from app.services.node_service import NodeService
# from app.models.node import NodeEntry
from app.services.demo_nodes import NODE_REGISTRY

router = APIRouter(prefix="/api", tags=["Nodes"])

@router.get("/ping")
def ping():
    return {"message": "Aegis API is alive!"}

@router.get("/nodes")
def get_nodes():
    return [
        {"node_id": node_id, "ip": data["ip"], "port": data["port"]}
        for node_id, data in NODE_REGISTRY.items()
    ]


# router = APIRouter(prefix="/api", tags=["Nodes"])

# @router.get("/nodes", response_model=list[NodeEntry])
# def get_nodes():
#     return NodeService.get_all_nodes()

# @router.get("/ping")
# def ping():
#     return {"message": "Aegis API is alive!"}
