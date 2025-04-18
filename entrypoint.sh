uv run ray start --head --dashboard-host 0.0.0.0
uv run --no-dev uvicorn api:app --host 0.0.0.0 --port 8080 --reload