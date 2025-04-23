uv run ray start --head --dashboard-host 0.0.0.0 -dashboard-port ${RAY_DASHBOARD_PORT:-8265}
uv run --no-dev uvicorn api:app --host 0.0.0.0 --port ${CONTAINER_PORT:-8080} --reload