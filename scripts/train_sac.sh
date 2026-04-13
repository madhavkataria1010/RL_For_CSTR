#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" -m src.training.train_modern_rl \
  --method sac \
  --config configs/base.yaml configs/execution/debug_local.yaml configs/methods/sac.yaml configs/scenarios/nominal.yaml
