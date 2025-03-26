import argparse
import time

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel
from p2pfl.node import Node
from p2pfl.utils.utils import set_test_settings

set_test_settings()


def __get_args():
    parser = argparse.ArgumentParser(description="P2PFL Node1 (active starter node)")
    parser.add_argument("--host", type=str, required=True, help="Host IP of this node")
    parser.add_argument("--port", type=int, required=True, help="Port to bind")
    parser.add_argument("--wait_peers", type=int, default=1, help="Number of peers to wait before starting learning")
    return parser.parse_args()


def node1(host: str, port: int, wait_peers: int):
    address = f"{host}:{port}"
    print(f"Node1 starting at {address}")

    node = Node(
        LightningModel(MLP()),
        P2PFLDataset.from_huggingface("p2pfl/MNIST"),
        address=address,
        learner=LightningLearner
    )

    node.start()

    print(f"Waiting for {wait_peers} peers...")
    while len(node.get_neighbors().keys()) < wait_peers:
        print(f"🔗 Known peers so far: {node.get_neighbors().keys()}")
        time.sleep(7)

    print("Enough peers connected. Starting federated learning")
    node.set_start_learning(rounds=2, epochs=1)

    while True:
        time.sleep(1)
        if node.state.round is None:
            break

    print("Node1 finished. Stopping.")
    node.stop()


if __name__ == "__main__":
    args = __get_args()
    node1(args.host, args.port, args.wait_peers)
