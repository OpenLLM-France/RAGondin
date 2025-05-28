from collections import Counter

import ray
from config.config import load_config
from fastapi import APIRouter

# load config
config = load_config()

# Create an APIRouter instance
router = APIRouter()

task_state_manager = ray.get_actor("TaskStateManager", namespace="ragondin")


indexer_queue = ray.get_actor("IndexerQueue", namespace="ragondin")
task_state_manager = ray.get_actor("TaskStateManager", namespace="ragondin")


@router.get("/info")
async def get_queue_info():
    # Get task states
    all_states: dict = await task_state_manager.get_all_states.remote()
    status_counts = Counter(all_states.values())

    # Structure status counts
    active_statuses = ["QUEUED", "SERIALIZING", "CHUNKING", "INSERTING"]
    active = {status: status_counts.get(status, 0) for status in active_statuses}
    total_active = sum(active.values())
    total_completed = status_counts.get("COMPLETED", 0)
    total_failed = status_counts.get("FAILED", 0)

    # Get worker info
    worker_info = await indexer_queue.pool_info.remote()
    total_workers = worker_info["total"]
    available_workers = worker_info["idle"]

    return {
        "workers": {"total": total_workers, "available": available_workers},
        "tasks": {
            "active": total_active,
            "active_statuses": active,
            "total_completed": total_completed,
            "total_failed": total_failed,
        },
    }
