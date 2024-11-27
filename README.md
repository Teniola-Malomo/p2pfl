![GitHub Logo](https://raw.githubusercontent.com/pguijas/p2pfl/main/other/logo.png)

# P2PFL - Federated Learning over P2P networks

[![GitHub license](https://img.shields.io/github/license/pguijas/federated_learning_p2p)](https://github.com/pguijas/p2pfl/blob/main/LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/pguijas/federated_learning_p2p)](https://github.com/pguijas/p2pfl/issues)
![GitHub contributors](https://img.shields.io/github/contributors/pguijas/federated_learning_p2p)
![GitHub forks](https://img.shields.io/github/forks/pguijas/federated_learning_p2p)
![GitHub stars](https://img.shields.io/github/stars/pguijas/federated_learning_p2p)
![GitHub activity](https://img.shields.io/github/commit-activity/m/pguijas/federated_learning_p2p)
[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fpguijas%2Fp2pfl%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/pguijas/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://join.slack.com/t/p2pfl/shared_invite/zt-2lbqvfeqt-FkutD1LCZ86yK5tP3Duztw)

P2PFL is a general-purpose open-source library designed for the execution (simulated and in real environments) of Decentralized Federated Learning systems, specifically making use of P2P networks and the gossip protocols.

## ✨ Key Features

P2PFL offers a range of features designed to make decentralized federated learning accessible and efficient. For detailed information, please refer to our [documentation](https://p2pfl.github.io/p2pfl/).

| Feature          | Description                                      |
|-------------------|--------------------------------------------------|
| 🚀 Easy to Use   | [Get started](https://p2pfl.github.io/p2pfl/quickstart.html) quickly with our intuitive API.       |
| 🛡️ Reliable     | Built for fault tolerance and resilience.       |
| 🌐 Scalable      | Leverages the power of peer-to-peer networks.    |
| 🧪 Versatile     | Experiment in simulated or real-world environments.|
| 🔒 Private       | Prioritizes data privacy with decentralized architecture.|
| 🧩 Flexible      | Integrate with PyTorch and TensorFlow (coming soon!).|
| 📈 Real-time Monitoring | Manage and track experiment through [P2PFL Web Services platform](https://p2pfl.com). | 
| 🧠 Model Agnostic | Use any machine learning model you prefer (e.g., PyTorch or Keras models). |
| 📡 Communication Protocol Agnostic | Choose the communication protocol that best suits your needs (e.g., gRPC). |

## 📥 Installation

### 👨🏼‍💻 For Users

```bash
pip install "p2pfl[torch]"
```

### 👨🏼‍🔧 For Developers

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/p2pfl/p2pfl/tree/develop?quickstart=1)

#### 🐍 Python (using Poetry)

```bash
git clone https://github.com/pguijas/p2pfl.git
cd p2pfl
poetry install -E torch 
```

> **Note:** Use the extras (`-E`) flag to install specific dependencies (e.g., `-E torch`). Use `--no-dev` to exclude development dependencies.

#### 🐳 Docker

```bash
docker build -t p2pfl .
docker run -it --rm p2pfl bash
```

## 🎬 Quickstart

To start using P2PFL, follow our [quickstart guide](https://pguijas.github.io/p2pfl/quickstart.html) in the documentation.

## 📚 Documentation & Resources

* **Documentation:** [https://pguijas.github.io/p2pfl/](https://p2pfl.github.io/p2pfl)
* **End-of-Degree Project Report:** [other/memoria.pdf](other/memoria.pdf)
* **Open Source Project Award Report:** [other/memoria-open-source.pdf](other/memoria-open-source.pdf)

## 🤝 Contributing

We welcome contributions! See `CONTRIBUTING.md` for guidelines. Please adhere to the project's code of conduct in `CODE_OF_CONDUCT.md`.

## 💬 Community

Connect with us and stay updated:

* [**GitHub Issues:**](https://github.com/p2pfl/p2pfl/issues) - For reporting bugs and requesting features.
* [**Google Group:**](https://groups.google.com/g/p2pfl) - For discussions and announcements.
* [**Slack:**](https://join.slack.com/t/p2pfl/shared_invite/zt-2lbqvfeqt-FkutD1LCZ86yK5tP3Duztw) - For real-time conversations and support.


## ⭐ Star History

A big thank you to the community for your interest in P2PFL! We appreciate your support and contributions.

[![Star History Chart](https://api.star-history.com/svg?repos=pguijas/p2pfl&type=Date)](https://star-history.com/#pguijas/p2pfl&Date)

## 📜 License

[GNU General Public License, Version 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
