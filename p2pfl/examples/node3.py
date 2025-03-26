"""
Example of a P2PFL MNIST node using a MLP model and a MnistFederatedDM.

This node will be connected to node1 and participate in the federated learning process.
"""

import argparse
import time

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel
from p2pfl.node import Node
from p2pfl.utils.utils import set_test_settings

set_test_settings()

def __get_args():
    parser = argparse.ArgumentParser(description="P2PFL Node3 (peer node)")
    parser.add_argument("--host", type=str, required=True, help="Host IP of this node")
    parser.add_argument("--port", type=int, required=True, help="Port to bind")
    parser.add_argument("--connect", type=str, required=True, help="Node1 address to connect to (ip:port)")
    return parser.parse_args()

def node3(host: str, port: int, connect_to: str):
    address = f"{host}:{port}"
    print(f"Node3 starting at {address}")

    node = Node(
        LightningModel(MLP()),
        P2PFLDataset.from_huggingface("p2pfl/MNIST"),
        address=address,
        learner=LightningLearner
    )

    node.start()

    print(f"🔗 Connecting to {connect_to}")
    node.connect(connect_to)
    time.sleep(4)

    print("🚀 Start learning")
    node.set_start_learning(rounds=2, epochs=1)

    while True:
        time.sleep(1)
        if node.state.round is None:
            break

    print("Node3 stopping")
    node.stop()

if __name__ == "__main__":
    args = __get_args()
    node3(args.host, args.port, args.connect)
