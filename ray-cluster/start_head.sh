#!/bin/bash
uv run ray start --head --dashboard-host 0.0.0.0 --dashboard-port ${RAY_DASHBOARD_PORT:-8265}
tail -f /dev/null
