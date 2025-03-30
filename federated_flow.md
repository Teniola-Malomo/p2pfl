# 🤝 Federated Flow: A Simple Guide to Setting Up P2PFL Locally with Docker Compose

Welcome to **Federated Flow**! This walkthrough will guide you through setting up a peer-to-peer federated learning system using [P2PFL](https://github.com/pguijas/p2pfl) — a framework for decentralized ML with PyTorch Lightning.

We'll use Docker Compose to run **three nodes** that collaborate on training an MNIST model using federated learning. Let’s keep it simple, smooth, and reproducible.

---

## 🛠️ What You Need

- Docker & Docker Compose installed
- Python (optional, only needed if testing scripts locally)
- Internet connection to download dependencies

---

## 📦 Step 1: Clone the Project

```bash
git clone https://github.com/p2pfl/p2pfl
cd p2pfl
```

> You should now be inside the `p2pfl` folder where all the code lives.

---

## 🧪 Step 2: Build the Docker Image

This step packages everything needed into a container image.

```bash
docker build -t p2pfl-deploy .
```

- `-t p2pfl-deploy` gives the image a name
- `.` means "build from this directory"

---

## 📄 Step 3: Add the Compose File

Create a file called `docker-compose.yml` in the root of the project:

```yaml
version: "3.8"

services:
  node1:
    image: p2pfl-deploy
    container_name: node1
    restart: unless-stopped
    networks:
      p2pfl-net:
        ipv4_address: 172.20.0.2
    command: >
      python p2pfl/examples/node1.py
      --host 172.20.0.2
      --port 5001
      --wait_peers 2

  node2:
    image: p2pfl-deploy
    container_name: node2
    restart: unless-stopped
    depends_on:
      - node1
    networks:
      p2pfl-net:
        ipv4_address: 172.20.0.3
    command: >
      python p2pfl/examples/node2.py
      --host 172.20.0.3
      --port 5002
      --connect 172.20.0.2:5001

  node3:
    image: p2pfl-deploy
    container_name: node3
    restart: unless-stopped
    depends_on:
      - node1
    networks:
      p2pfl-net:
        ipv4_address: 172.20.0.4
    command: >
      python p2pfl/examples/node3.py
      --host 172.20.0.4
      --port 5003
      --connect 172.20.0.2:5001

networks:
  p2pfl-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

---

## 📂 Step 4: Add node3.py

Copy `node2.py` to `node3.py` and modify it slightly to reflect that it's node 3:

```bash
cp p2pfl/examples/node2.py p2pfl/examples/node3.py
```

Edit the `print` statements and function names in `node3.py` so logs will show "Node3" instead of "Node2". Example:

```python
print(f"🟢 Node3 starting at {address}")
```

---

## 🚀 Step 5: Run the System

Now it's time to launch everything:

```bash
docker-compose up --build
```

You’ll start seeing logs from node1, node2, and node3 as they:

- Start up
- Connect to each other
- Vote and elect a training set
- Train and evaluate the model

---

## 🧹 Optional: Stop and Clean Up

```bash
docker-compose down
```

This will stop the containers and remove the network.

To also remove volumes:

```bash
docker-compose down -v
```

---

## 🧠 Recap

| Node  | IP              | Role             |
| ----- | --------------- | ---------------- |
| node1 | 172.20.0.2:5001 | main coordinator |
| node2 | 172.20.0.3:5002 | federated peer   |
| node3 | 172.20.0.4:5003 | federated peer   |

All nodes share the MNIST dataset, talk via gRPC, and train in rounds.

---

## 🌐 What Next?

- Deploy across multiple machines or cloud servers (GCP, AWS)
- Use a different dataset or model
- Visualize results with TensorBoard or Grafana
- Add secure TLS certificates to your gRPC layer

---

Need help scaling to Google Cloud or Docker Swarm? Just ask! Let's keep your federated flow running strong.

Happy hacking 🧠💻!

