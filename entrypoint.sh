uv run ray start --head --metrics-export-port=8081 --dashboard-host 0.0.0.0 --num-gpus=1
uv run uvicorn api:app --host 0.0.0.0 --port 8080 --reload