#!/bin/bash
uv run ray start --address ${HEAD_NODE_IP:-10.0.0.1}:6379
tail -f /dev/null