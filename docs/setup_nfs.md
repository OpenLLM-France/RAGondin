# 📡 NFS Setup for Shared Storage (Ray Cluster)

In a Ray distributed setup, **all worker nodes need access to certain shared resources** used by the application.  
This includes:

- `.env` (environment variables for models and settings)
- `.hydra_config` (application configuration)
- SQLite DB (`/volumes`)
- Uploaded files (`/data`)
- Model weights (e.g. `/model_weights` if using HF local cache)

---

## 1️⃣ Setup VPN (if required)

If your Ray nodes are **not on the same local network**, set up a VPN between them first.  
➡ Refer to the dedicated [VPN setup guide](../docs/setup_vpn.md).  
You can skip this step if your nodes are already on the same LAN.

---

## 2️⃣ Setup NFS (Network File System)

This allows your **main machine (Ray head + API host)** to share directories with other worker nodes.

### Install NFS

#### On the **main machine (server)**
```bash
sudo apt update
sudo apt install -y nfs-kernel-server
```

#### On all **worker nodes (clients)**
```bash
sudo apt update
sudo apt install -y nfs-common
```

---

### Setup NFS server on the main machine

Create the shared folder:
```bash
sudo mkdir -p /ray_mount
```

Edit the NFS exports file:
```bash
sudo nano /etc/exports
```

Add this line:
```
/ray_mount 192.168.42.0/24(rw,sync,no_subtree_check)
```
> ⚠ Replace `192.168.42.0/24` with your network's actual subnet.

Apply the exports and restart the service:
```bash
sudo exportfs -a
sudo systemctl restart nfs-kernel-server
```

---

### Mount NFS share on worker nodes

Mount the shared folder:
```bash
sudo mkdir -p /ray_mount
sudo mount -t nfs 192.168.42.226:/ray_mount /ray_mount
```
> ⚠ Replace `192.168.42.226` with the **IP of the main machine**.

To make the mount permanent after reboot:
```bash
echo "192.168.42.226:/ray_mount /ray_mount nfs defaults 0 0" | sudo tee -a /etc/fstab
```

---

### Copy required data to the shared folder

From the main machine:
```bash
sudo cp -r .hydra_config /ray_mount/
sudo cp .env /ray_mount/
sudo mkdir /ray_mount/volumes /ray_mount/data /ray_mount/model_weights
sudo chown -R ubuntu:ubuntu /ray_mount
```
> ✅ Ensure that the ownership is set to the user running Ray workers (e.g. `ubuntu`) so that all nodes can read/write.

---

Now, all Ray nodes will have **consistent access to required data and configurations** via `/ray_mount`.

