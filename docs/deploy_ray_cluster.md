# ⚡ Distributed Deployment in a Ray Cluster

This guide explains how to deploy **RAGondin** across multiple machines using **Ray** for distributed indexing and processing.

---

## ✅ 1. Set Environment Variables

Ensure your `.env` file includes the standard app variables **plus Ray-specific ones** listed below:

```env
# Ray runtime
RAY_DASHBOARD_PORT=8265
RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Ray cluster-specific
SHARED_ENV=/ray_mount/.env
LOCAL_DEPLOYMENT=false
RAY_ADDRESS=ray://<HEAD_NODE_IP>:10001

# Resource allocation per actor
RAY_NUM_CPUS=4
RAY_NUM_GPUS=0.5
```

> 🧠 **Tip**: The last two variables define **per-actor resource usage**.  
> For example, if one indexation consumes ~7GB of VRAM and your GPU has 16GB, setting `RAY_NUM_GPUS=0.5` lets you run **2 indexers per node**. In a 2-node cluster, that means **4 concurrent indexation tasks**.

---

## 📁 2. Set Up Shared Storage

All nodes need to access shared configuration and data folders.  
We recommend using **NFS** for this.

➡ Follow the [NFS Setup Guide](./setup_nfs.md) to configure:

- Shared access to:
  - `.env`
  - `.hydra_config`
  - `/volumes` (SQLite)
  - `/data` (uploaded files)
  - `/model_weights` (embedding model cache)

---

## 🚀 3. Start the Ray Cluster

First, prepare your `cluster.yaml` file. Here's an example for a **local provider**:

```yaml
cluster_name: rag-cluster
provider:
  type: local
  head_ip: 10.0.0.1
  worker_ips: [10.0.0.2]  # Static IPs of other nodes (does not auto-start workers)

docker:
  image: ghcr.io/openllm-france/ragondin-ray
  pull_before_run: true
  container_name: ray_node
  run_options:
    - --gpus all
    - -v /ray_mount/model_weights:/app/model_weights
    - -v /ray_mount/data:/app/data
    - -v /ray_mount/volumes:/app/volumes
    - -v /ray_mount/.hydra_config:/app/.hydra_config
    - --env-file /ray_mount/.env

auth:
  ssh_user: ubuntu
  ssh_private_key: path/to/private/key # Replace with your actual ssh key path

head_setup_commands:
  - bash /app/ray-cluster/start_head.sh
```

> 🛠️ The base image (`ghcr.io/openllm-france/ragondin-ray`) must be built from `Dockerfile.ray` and pushed to a container registry before use.

### ⬆️ Launch the cluster

```bash
uv run ray up cluster.yaml
```

### ➕ Join the cluster from worker nodes

Run this on each worker node to start the Ray container:

```bash
docker run --rm -d \
  --gpus all \
  --network host \
  --env-file /ray_mount/.env \
  -v /ray_mount/model_weights:/app/model_weights \
  -v /ray_mount/data:/app/data \
  -v /ray_mount/volumes:/app/volumes \
  -v /ray_mount/.hydra_config:/app/.hydra_config \
  --name ray_node_worker \
  ghcr.io/openllm-france/ragondin-ray \
  bash /app/ray-cluster/start_worker.sh
```

---

## 🐳 4. Launch the RAGondin App

Use the Ray-compatible Docker Compose setup:

```bash
docker compose -f docker-compose-ray.yaml up -d
```

Once running, **RAGondin will auto-connect** to the Ray cluster using `RAY_ADDRESS` from `.env`.

---

With this setup, your app is now fully distributed and ready to handle concurrent tasks across your Ray cluster.


## 🛠️ Troubleshooting

### ❌ Permission Denied Errors

If you encounter errors like `Permission denied` when Ray or Docker tries to access shared folders (SQL database, model files, ...), it's likely due to insufficient permissions on the host system.

👉 To resolve this, you can set full read/write/execute permissions on the shared directory:

```bash
sudo chmod -R 777 /ray_mount
```