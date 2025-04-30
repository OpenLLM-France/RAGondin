uv run ray start --head --dashboard-host 0.0.0.0 --dashboard-port ${RAY_DASHBOARD_PORT:-8265}
ENV_ARG=""
if [[ -n "${SHARED_ENV}" ]]; then
  ENV_ARG="--env-file=${SHARED_ENV}"
fi
uv run --no-dev $ENV_ARG uvicorn api:app --host 0.0.0.0 --port ${APP_iPORT:-8080} --reload