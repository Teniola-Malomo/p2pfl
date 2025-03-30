from fastapi import APIRouter
from app.services.node_service import NodeService
from app.models.node import NodeEntry

router = APIRouter(prefix="/api", tags=["Nodes"])

@router.get("/nodes", response_model=list[NodeEntry])
def get_nodes():
    return NodeService.get_all_nodes()

@router.get("/ping")
def ping():
    return {"message": "Aegis API is alive!"}
