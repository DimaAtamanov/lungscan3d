#!/usr/bin/env bash
set -euo pipefail
HOST="${TENSORBOARD_HOST:-127.0.0.1}"
PORT="${TENSORBOARD_PORT:-6006}"
LOGDIR="${TENSORBOARD_LOGDIR:-lightning_logs}"
tensorboard --host "${HOST}" --port "${PORT}" --logdir "${LOGDIR}"
