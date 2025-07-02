#!/bin/bash
uv run ray start --head --dashboard-host 0.0.0.0 --dashboard-port ${RAY_DASHBOARD_PORT:-8265} --node-ip-address ${HEAD_NODE_IP}
tail -f /dev/null
