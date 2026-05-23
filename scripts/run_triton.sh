#!/usr/bin/env bash
set -euo pipefail
docker run --rm --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$(pwd)/triton_model_repository:/models" \
  nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models
