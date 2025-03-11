#!/bin/bash

PYTHON_FILE="manage_collection.py"

VALUES=("true" "false")

cd ragondin

for arg1 in "${VALUES[@]}"; do
    echo "Running $PYTHON_FILE with arguments: $arg1"
    uv run python manage_collection.py -f ../data -o loader.image_captioning=\'${arg1}\' -o vectordb.enable='false'
done
