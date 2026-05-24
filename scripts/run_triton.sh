#!/usr/bin/env bash
set -euo pipefail
MODEL_REPOSITORY="${MODEL_REPOSITORY:-$(pwd)/triton_model_repository}"
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.05-py3}"
docker run --rm --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${MODEL_REPOSITORY}:/models" \
  "${TRITON_IMAGE}" \
  tritonserver --model-repository=/models
