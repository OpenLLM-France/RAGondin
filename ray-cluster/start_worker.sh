#!/bin/bash
uv run ray start --address ${RAY_HEAD_ADDRESS:-10.0.0.1:6379}
tail -f /dev/null