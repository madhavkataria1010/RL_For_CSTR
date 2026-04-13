#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Project virtual environment not found at ${PYTHON_BIN}. Run scripts/setup.sh first." >&2
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p run_logs
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

run_paper_job() {
  local method_yaml="$1"
  local scenario_yaml="$2"
  local seed="$3"
  local module="$4"
  local tag="$5"
  local log_path="run_logs/${tag}.log"
  echo "[paper] launching ${tag}"
  (
    "${PYTHON_BIN}" -m "${module}" \
      --config \
      configs/base.yaml \
      configs/execution/results_repair.yaml \
      "${method_yaml}" \
      "${scenario_yaml}" \
      --seed "${seed}"
  ) >"${log_path}" 2>&1 &
}

run_gpu_job() {
  local method="$1"
  local method_yaml="$2"
  local seed="$3"
  local tag="$4"
  local log_path="run_logs/${tag}.log"
  echo "[gpu] launching ${tag}"
  (
    "${PYTHON_BIN}" -m src.training.train_modern_rl \
      --method "${method}" \
      --config \
      configs/base.yaml \
      configs/execution/results_repair.yaml \
      "${method_yaml}" \
      configs/scenarios/nominal.yaml \
      --seed "${seed}"
  ) >"${log_path}" 2>&1 &
}

SEEDS=(0 1 2 3 4)

for seed in "${SEEDS[@]}"; do
  run_paper_job configs/methods/pure_rl_paper.yaml configs/scenarios/nominal.yaml "${seed}" src.training.train_pure_rl_paper "pure_rl_nominal_seed${seed}"
  run_paper_job configs/methods/cirl_reproduced.yaml configs/scenarios/nominal.yaml "${seed}" src.training.train_cirl "cirl_nominal_seed${seed}"
  run_paper_job configs/methods/pure_rl_paper.yaml configs/scenarios/disturbance.yaml "${seed}" src.training.train_pure_rl_paper "pure_rl_disturbance_seed${seed}"
  run_paper_job configs/methods/cirl_reproduced.yaml configs/scenarios/disturbance.yaml "${seed}" src.training.train_cirl "cirl_disturbance_seed${seed}"
  run_paper_job configs/methods/cirl_highop_extended_paper.yaml configs/scenarios/highop.yaml "${seed}" src.training.train_cirl "cirl_highop_seed${seed}"
  run_paper_job configs/methods/dr_cirl.yaml configs/scenarios/nominal.yaml "${seed}" src.training.train_dr_cirl "dr_cirl_nominal_seed${seed}"
done

for seed in "${SEEDS[@]}"; do
  run_gpu_job sac configs/methods/sac.yaml "${seed}" "sac_nominal_seed${seed}"
done
for seed in "${SEEDS[@]}"; do
  run_gpu_job td3 configs/methods/td3.yaml "${seed}" "td3_nominal_seed${seed}"
done
for seed in "${SEEDS[@]}"; do
  run_gpu_job tqc configs/methods/tqc.yaml "${seed}" "tqc_nominal_seed${seed}"
done
for seed in "${SEEDS[@]}"; do
  run_gpu_job ppo configs/methods/ppo.yaml "${seed}" "ppo_nominal_seed${seed}"
done

echo "All repair training jobs launched concurrently."
wait

"${PYTHON_BIN}" -m src.evaluation.paper_reproduction \
  --config configs/base.yaml configs/execution/results_repair.yaml configs/scenarios/nominal.yaml > run_logs/eval_paper_nominal.log 2>&1
"${PYTHON_BIN}" -m src.evaluation.paper_reproduction \
  --config configs/base.yaml configs/execution/results_repair.yaml configs/scenarios/disturbance.yaml > run_logs/eval_paper_disturbance.log 2>&1
"${PYTHON_BIN}" -m src.evaluation.paper_reproduction \
  --config configs/base.yaml configs/execution/results_repair.yaml configs/scenarios/highop.yaml > run_logs/eval_paper_highop.log 2>&1
"${PYTHON_BIN}" -m src.evaluation.extended_benchmark \
  --config configs/base.yaml configs/execution/results_repair.yaml configs/scenarios/nominal.yaml > run_logs/eval_extended_results_repair.log 2>&1
"${PYTHON_BIN}" -m src.plotting.summary_tables > run_logs/make_figures_results_repair.log 2>&1

echo "results_repair run completed"
