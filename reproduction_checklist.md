# Reproduction Checklist

## Locked Sources

- [x] Paper title confirmed: **Control-Informed Reinforcement Learning for Chemical Processes**
- [x] arXiv source locked: `2408.13566`
- [x] journal DOI locked: `10.1021/acs.iecr.4c03233`
- [x] official repository locked: `OptiMaL-PSE-Lab/CIRL`
- [x] pinned reference commit locked: `c3fd22580a15d1e008570c08e78b38cdd887ef2c`
- [x] pinned commit message noted: `Implemented RGA`
- [x] official artifact folders identified:
  - `CIRL/`
  - `data/`
  - `plots/`

## Supplementary Material

- [x] No separate supplementary-material URL has been locked as a primary source of truth for this scaffold.
- [x] Reproduction is therefore anchored to:
  - the paper text
  - the official repository
  - the released weights/results/plot artifacts

## Reproduction Targets

- [x] Reproduce static PID baseline behavior
- [x] Reproduce `Pure-RL (paper baseline)`
- [x] Reproduce CIRL nominal tracking results
- [x] Reproduce CIRL high-operating-point evaluation
- [x] Reproduce disturbance rejection comparison
- [x] Recreate the official learning-curve figure
- [x] Recreate the official network-size figure
- [x] Record static PID gain artifacts
- [x] Record the RGA matrix output

## Official Files To Mirror In The New Pipeline

- [x] `CIRL/cstr_model.py`
- [x] `CIRL/cirl_policy.py`
- [x] `CIRL/training_CIRL.py`
- [x] `CIRL/static_pid_gains.py`
- [x] `CIRL/SP_vis_paper.py`
- [x] `CIRL/SP_vis_paper_highop.py`
- [x] `CIRL/learning_curve_vis.py`
- [x] `CIRL/dist_vis_presentation.py`
- [x] `CIRL/network_size_vis.py`
- [x] `CIRL/RGA.py`

## Paper-Faithful Methods That Must Stay Unchanged

- [x] `Static PID`
- [x] `Pure-RL (paper baseline)`
- [x] `CIRL reproduced`
- [x] `CIRL high-op extended (paper reproduction only)`

## Project Contribution Rule

- [x] `DR-CIRL` is the only project contribution.
- [x] `DR-CIRL` keeps:
  - the reproduced CIRL architecture
  - the same optimizer family
  - the same reward/cost
  - the same observation design
- [x] The only planned modification is domain randomization during training.

## Standardized Modern RL Baselines

- [x] Primary comparator: `SAC`
- [x] Strong comparators: `TD3`, `TQC`
- [x] Broader reference baseline: `PPO`
- [x] These methods are added for broader comparison only; they do not replace any paper method.

## GRPO Scope Control

- [x] `GRPO` is not part of the core benchmark by default.
- [x] If not implemented, the repo must state:
  - `GRPO was excluded from the core benchmark because it is not a standard continuous-control baseline for this task`

## Paper-Vs-Code Mismatches To Document Explicitly

- [x] Reward naming mismatch:
  - environment returns cost-like positive value
  - plotting scripts report `-cost` as reward
- [x] Time-axis mismatch:
  - environment uses `linspace(0, 100, ns)`
  - paper-style plots use `linspace(0, 25, ns)`
- [x] Action-bound mismatch:
  - direct-action RL maps `T_c` to `[290, 400]`
  - PID execution clamps `T_c` to `[290, 450]`
- [x] Disturbance-training values in code:
  - `[1.7, 1.6, 1.9]`
- [x] Policy implementation detail:
  - `n_layers` exists in the constructor
  - additional layers are not used in the forward pass
- [x] Static PID tuning script uses `test=True` with `ns = 360`

## Exact vs Approximate Reference Values

Use this rule in `results/tables/reproduction_summary.csv`:

- mark a value as `exact` when it comes directly from:
  - a paper table
  - a released official artifact
  - a clearly printed official script output
- mark a value as `approximate` when it is visually estimated from a figure

Current locked scalar references:

- [x] nominal reward table values treated as exact
  - `Pure-RL (paper baseline): -2.08`
  - `CIRL reproduced: -1.33`
  - `Static PID: -1.77`
- [x] high-op reward table values treated as exact
  - `CIRL initial: -4.04`
  - `CIRL extended: -2.07`
  - `Static PID: -6.81`
- [x] disturbance reward values treated as exact
  - `Pure-RL (paper baseline): -1.76`
  - `CIRL reproduced: -1.38`

## Execution Schedule Lock

- [x] Local debug: `1` seed, nominal only
- [x] CUDA validation: `3` seeds, nominal + one uncertainty scenario
- [x] Final CUDA benchmark: `5` seeds for benchmark tables
- [x] `10` seeds only where required for paper-style learning curves

## Execution Priority Lock

- [x] After faithful paper reproduction:
  1. `SAC`
  2. `TD3`
  3. `DR-CIRL`
  4. `TQC`
  5. `PPO`

If compute becomes tight:

- [x] keep `SAC`
- [x] keep `TD3`
- [x] keep `DR-CIRL`
- [x] try `TQC` if feasible
- [x] keep `PPO` as the lowest-priority baseline

## Fairness Rules For Final Benchmarking

- [x] same environment dynamics per scenario
- [x] same observation channels per method where applicable
- [x] same action bounds
- [x] same evaluation horizon
- [x] same initial conditions
- [x] same disturbance schedules
- [x] same uncertainty schedules per seed
- [x] same total environment steps
- [x] same evaluation frequency
- [x] same number of evaluation episodes
- [x] same model selection rule
- [x] same final evaluation protocol
- [x] deterministic policy inference wherever applicable in final benchmarking
- [x] PID tuned once on nominal setup and then frozen
- [x] no per-scenario retuning after benchmark definitions freeze

## Domain Randomization Freeze For DR-CIRL

- [x] training randomization parameters:
  - `C_A,in`
  - `UA`
  - shared `k0` kinetic-group scaling unless implementation forces `E_over_R`
- [x] training range: `±10%`
- [x] stress-test ranges: `±10%` and `±20%`
- [x] sampling distribution: uniform around nominal

## Success-Criteria Lock

- [x] Primary success criterion:
  - `DR-CIRL` must improve uncertainty robustness relative to reproduced CIRL without materially degrading nominal tracking
- [x] Secondary success criterion:
  - `DR-CIRL` should be competitive with or better than the strongest modern RL comparator, especially `SAC` or `TD3`, on robustness-focused metrics
- [x] It is **not** enough to beat only `Pure-RL (paper baseline)`.

## No-Substitution Rule

- [x] Do not replace the paper baseline with modern RL algorithms.
- [x] Do not rename the paper baseline to vanilla RL.
- [x] Do not claim an extension improvement without multi-seed evidence.
