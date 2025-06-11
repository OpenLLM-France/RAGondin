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
DATA_VOLUME = /ray_mount/data
MODEL_WEIGHTS_VOLUME = /ray_mount/model_weights
RAY_ADDRESS=ray://<HEAD_NODE_IP>:10001

# Worker pool settings
RAY_POOL_SIZE=8
RAY_MAX_TASKS_PER_WORKER=5 # Worker restarts after 5 tasks to avoid memory leak

# Resource requirements per indexation task
RAY_NUM_GPUS=0.5
```

✅ Use host IPs instead of Docker service names :

- EMBEDDER_BASE_URL=http://<HOST-IP>:8000/v1  # ✅ instead of http://vllm:8000/v1
- VDB_HOST=<HOST-IP>                          # ✅ instead of VDB_HOST=milvus


> 🧠 **Tips**  
>
> - `RAY_NUM_GPUS` defines **per-actor resource requirements**. Ray will not start a task until these resources are available on one of the nodes.  
>   For example, if one indexation consumes ~1GB of VRAM and your GPU has 4GB, setting `RAY_NUM_GPUS=0.25` allows you to run **4 indexers per node**. In a 2-node cluster, that means up to **8 concurrent indexation tasks**.  
>
> - `RAY_POOL_SIZE` defines the number of worker actors that will be created to handle indexation tasks. It acts like a **maximum concurrency limit**.  
>   Using the previous example, you can set `POOL_SIZE=8` to fully utilize your cluster capacity.  
>   ⚠️ If other GPU-intensive services are running on your nodes (e.g. vLLM, the RAG API), make sure to **reserve enough GPU memory** for them and subtract that from your total when calculating the safe pool size.

---

## 📁 2. Set Up Shared Storage

All nodes need to access shared configuration and data folders.  
We recommend using **NFS** for this.

➡ Follow the [NFS Setup Guide](./setup_nfs.md) to configure:

- Shared access to:
  - `.env`
  - `.hydra_config`
  - `/db` (SQLite)
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
    - -v /ray_mount/db:/app/db
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
  -v /ray_mount/db:/app/db \
  -v /ray_mount/.hydra_config:/app/.hydra_config \
  --name ray_node_worker \
  ghcr.io/openllm-france/ragondin-ray \
  bash /app/ray-cluster/start_worker.sh
```

---

## 🐳 4. Launch the RAGondin App

Use the Docker Compose setup:

```bash
docker compose up -d
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