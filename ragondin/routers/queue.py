from collections import Counter

import ray
from config.config import load_config
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

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


@router.get("/tasks", name="list_tasks")
async def list_tasks(request: Request, task_status: str | None = None):
    """
    GET /tasks
      - ?status=active    → all tasks whose status ∈ {QUEUED, SERIALIZING, CHUNKING, INSERTING}
      - ?status=task_status   → all tasks whose status == task_status (not case-sensitive)
      - (no status param) → all tasks
    """
    # Retrieve every task_id → status
    all_states: dict[str, str] = await task_state_manager.get_all_states.remote()

    # Determine which IDs to include based on the `status` query param
    if task_status is None:
        # No filter: include all task IDs
        filtered_ids = list(all_states.keys())

    else:
        # If status=active, treat as a special “umbrella” of multiple statuses
        if task_status.lower() == "active":
            active_statuses = {"QUEUED", "SERIALIZING", "CHUNKING", "INSERTING"}
            filtered_ids = [
                task_id for task_id, st in all_states.items() if st in active_statuses
            ]
        else:
            # Filter by exact match of the status string (case‐sensitive)
            filtered_ids = [
                task_id
                for task_id, st in all_states.items()
                if st.lower() == task_status.lower()
            ]

    # Build a list of {"link": "<URL to GET /tasks/{task_id}>"}
    tasks: list[dict[str, str]] = [
        {"link": str(request.url_for("get_task_status", task_id=task_id))}
        for task_id in filtered_ids
    ]

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"tasks": tasks},
    )
