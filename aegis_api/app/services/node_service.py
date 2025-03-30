from p2pfl.communication.protocols.neighbors import Neighbors
from app.models.node import NodeEntry

class NodeService:
    neighbors = Neighbors(self_addr="")
    
    @classmethod
    def get_all_nodes(cls) -> list[NodeEntry]:
        nodes_dict = cls.neighbors.get_all()
        return [
            NodeEntry(node_id=node_id, metadata=meta)
            for node_id, meta in nodes_dict.items()
        ]
