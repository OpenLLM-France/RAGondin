uv run ray start --head --dashboard-host 0.0.0.0 -dashboard-port ${RAY_DASHBOARD_PORT:-8265}
ENV_ARG=""
if [[ -n "${UV_ENV_FILE}" ]]; then
  ENV_ARG="--env-file=${UV_ENV_FILE}"
fi
uv run --no-dev $ENV_ARG uvicorn api:app --host 0.0.0.0 --port ${CONTAINER_PORT:-8080} --reload