#!/usr/bin/env bash
set -euo pipefail

python3 -m src.training.train_pid \
  --config configs/base.yaml configs/execution/debug_local.yaml configs/methods/static_pid.yaml configs/scenarios/nominal.yaml

