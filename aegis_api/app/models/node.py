from pydantic import BaseModel
from typing import Any

class NodeEntry(BaseModel):
    node_id: str
    metadata: Any  # Later you can replace this with a typed model
