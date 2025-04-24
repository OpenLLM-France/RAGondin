#!/bin/bash

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

uv run ray start --head --dashboard-host 0.0.0.0
uv run --no-dev uvicorn api:app --host 0.0.0.0 --port ${CONTAINER_PORT:-8080} --reload