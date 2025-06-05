from collections import Counter

import ray
from config.config import load_config
from fastapi import APIRouter

# load config
config = load_config()

# Create an APIRouter instance
router = APIRouter()

task_state_manager = ray.get_actor("TaskStateManager", namespace="ragondin")


serializer_queue = ray.get_actor("SerializerQueue", namespace="ragondin")
task_state_manager = ray.get_actor("TaskStateManager", namespace="ragondin")


def _format_pool_info(worker_info: dict[str, int]) -> dict[str, int]:
    """
    Convert SerializerQueue.pool_info() output into a concise dict for the API.
    """
    return {
        "total_slots": worker_info["total_capacity"],
        "free_slots": worker_info["free_slots"],
        "busy_slots": worker_info["current_load"],
        "pool_size": worker_info["pool_size"],
        "max_per_actor": worker_info["max_tasks_per_worker"],
    }


@router.get("/info")
async def get_queue_info():
    all_states: dict = await task_state_manager.get_all_states.remote()
    status_counts = Counter(all_states.values())

    active_statuses = ["QUEUED", "SERIALIZING", "CHUNKING", "INSERTING"]
    active = {s: status_counts.get(s, 0) for s in active_statuses}

    task_summary = {
        "active": sum(active.values()),
        "active_statuses": active,
        "total_completed": status_counts.get("COMPLETED", 0),
        "total_failed": status_counts.get("FAILED", 0),
    }

    worker_info = await serializer_queue.pool_info.remote()
    workers_block = _format_pool_info(worker_info)

    return {"workers": workers_block, "tasks": task_summary}
