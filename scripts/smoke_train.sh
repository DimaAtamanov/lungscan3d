#!/usr/bin/env bash
set -euo pipefail
lungscan3d train data=synthetic trainer.max_epochs=3 logging.mode=none "$@"
