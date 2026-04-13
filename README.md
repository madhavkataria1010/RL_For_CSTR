# Safe and Robust Control-Informed Reinforcement Learning for Nonlinear Chemical Process Control

This repository implements a reproducible reproduction-and-extension pipeline around **Control-Informed Reinforcement Learning for Chemical Processes** on the nonlinear CSTR benchmark. The project keeps the paper reproduction faithful, adds standardized modern RL baselines for broader comparison, and reserves **DR-CIRL** as the only new contribution.

## What Is Implemented

- paper-faithful CSTR environment with explicit `paper_exact` and standardized benchmark modes
- paper reproduction controllers:
  - `Static PID`
  - `Pure-RL (paper baseline)`
  - `CIRL reproduced`
  - `CIRL high-op extended (paper reproduction only)` via the shared CIRL training path
- standardized modern RL wrappers:
  - `SAC`
  - `TD3`
  - `TQC`
  - `PPO`
- `DR-CIRL` with frozen domain randomization over `C_A,in`, `UA`, and a shared `k0` scale
- config-driven training, evaluation, plotting, manifests, trajectory logging, and tests

## Source Of Truth

- Paper (arXiv): `2408.13566`
- Paper (journal DOI): `10.1021/acs.iecr.4c03233`
- Official code repository: `OptiMaL-PSE-Lab/CIRL`
- Locked reference commit: `c3fd22580a15d1e008570c08e78b38cdd887ef2c`

Primary notes:

- [paper_notes.md](/Users/madhav/Developer/RL_For_CSTR/paper_notes.md)
- [reproduction_checklist.md](/Users/madhav/Developer/RL_For_CSTR/reproduction_checklist.md)

## Repo Layout

```text
configs/
  execution/
  methods/
  scenarios/
results/
  figures/
  processed/
  raw/
  tables/
scripts/
src/
tests/
paper_notes.md
reproduction_checklist.md
summary_for_student.md
presentation_points.md
```

## Setup

The setup helper now creates a local virtual environment at `.venv`, installs dependencies there, and downloads the pinned official reference archive into `/tmp/CIRL_ref`.
On Linux hosts with NVIDIA GPUs, it also installs a CUDA-compatible PyTorch wheel explicitly instead of relying on PyPI's default wheel selection.

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Conda is also supported:

```bash
conda env create -f environment.yml
conda activate rl-cstr
```

## Local Validation

Run the test suite:

```bash
pytest -q
```

Smoke-test training scripts:

```bash
bash scripts/train_pid.sh
bash scripts/train_pure_rl_paper.sh
bash scripts/train_cirl.sh
bash scripts/train_sac.sh
```

Smoke-test evaluation and figures:

```bash
bash scripts/eval_paper_reproduction.sh
bash scripts/eval_extended_benchmark.sh
bash scripts/make_figures.sh
```

## Staged Execution Schedule

1. Local debug
   - `1` seed
   - nominal scenario only
   - smoke-test horizon
2. CUDA validation
   - `3` seeds
   - nominal plus one uncertainty scenario
3. Final CUDA benchmark
   - `5` seeds for benchmark tables
4. Paper-style learning curves
   - `10` seeds only where required

## Practical Execution Priority

1. faithful paper reproduction
2. `SAC`
3. `TD3`
4. `DR-CIRL`
5. `TQC`
6. `PPO`

If compute becomes tight:

- keep `SAC`
- keep `TD3`
- keep `DR-CIRL`
- try `TQC` if feasible
- keep `PPO` as lowest priority

## A5000 Execution Path

The repo includes `Agent.md` with the remote path and GPU instructions. The intended execution path is:

```bash
rsync -av --delete ./ root@10.36.16.15:/madhav/RL_For_CSTR/
ssh root@10.36.16.15
cd /madhav/RL_For_CSTR
python3 -m venv /madhav/venvs/rl-cstr
source /madhav/venvs/rl-cstr/bin/activate
bash scripts/setup.sh
pytest -q
```

Then move from debug to validation:

```bash
python -m src.training.train_modern_rl \
  --method sac \
  --config configs/base.yaml configs/execution/cuda_validation.yaml configs/methods/sac.yaml configs/scenarios/nominal.yaml

python -m src.training.train_modern_rl \
  --method td3 \
  --config configs/base.yaml configs/execution/cuda_validation.yaml configs/methods/td3.yaml configs/scenarios/nominal.yaml

python -m src.training.train_dr_cirl \
  --config configs/base.yaml configs/execution/cuda_validation.yaml configs/methods/dr_cirl.yaml configs/scenarios/nominal.yaml
```

## Outputs

The pipeline writes:

- raw run artifacts and trajectories under `results/raw/`
- processed report snippets under `results/processed/`
- tables under `results/tables/`
- publication-style figures under `results/figures/`

Important generated tables:

- `results/tables/reproduction_summary.csv`
- `results/tables/modern_rl_baselines.csv`
- `results/tables/master_comparison_table.csv`
- `results/tables/average_rank_table.csv`

## Notes

- `GRPO` is not part of the core benchmark and should be reported as exploratory only if ever attempted.
- Final benchmark claims should come from CUDA runs, not local Apple Silicon debug runs.
- `DR-CIRL` is the only project contribution; all other additions are comparison baselines or reproduction artifacts.
