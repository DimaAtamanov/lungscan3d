#!/usr/bin/env bash
set -euo pipefail
HOST="${MLFLOW_HOST:-127.0.0.1}"
PORT="${MLFLOW_PORT:-8080}"
mlflow ui --host "${HOST}" --port "${PORT}"
