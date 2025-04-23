uv run ray start --head --dashboard-host 0.0.0.0
uv run --no-dev --env-file=/ray_mount/.env uvicorn api:app --host 0.0.0.0 --port 8080 --reload 