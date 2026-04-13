# Presentation Points

## Problem Statement

- Nonlinear chemical process control remains difficult when the plant is nonlinear, disturbances are present, and operating conditions shift.
- Pure model-free RL can be flexible but often struggles with robustness and generalization.
- Classical PID is robust and interpretable but limited for highly nonlinear operating regimes.
- The project studies whether **control-informed RL** can combine the strengths of both.

## Reference Method

- The reference paper embeds PID structure into the policy through a CIRL architecture.
- The official benchmark plant is a nonlinear CSTR with regulation of:
  - product concentration `C_B`
  - reactor volume `V`
- Manipulated variables are:
  - cooling temperature `T_c`
  - inlet flow rate `F_in`

## Paper Reproduction Methods

- `Static PID`
- `Pure-RL (paper baseline)`
- `CIRL reproduced`
- `CIRL high-op extended (paper reproduction only)`

## Added Standardized Baselines

- `SAC`
- `TD3`
- `TQC`
- `PPO`

Framing:

- these are added only for broader modern continuous-control comparison
- they do not replace or redefine the paper reproduction
- `SAC` is the primary modern RL comparator

## Project Contribution

- `DR-CIRL`

Definition:

- same CIRL architecture
- same optimizer family
- same reward/cost
- same observation design
- only change is domain randomization during training

## Benchmark Suite

Tier 1:

- nominal tracking
- disturbance rejection
- high operating point
- parametric uncertainty

Tier 2 if time/compute allow:

- measurement noise
- actuator saturation
- unseen setpoints

## Metrics

- tracking quality:
  - IAE
  - ISE
  - ITAE
  - RMSE
- transient performance:
  - overshoot
  - settling time
- control quality:
  - control variation
- robustness and safety:
  - constraint violations
  - recovery time
- overall comparison:
  - average rank

## Fairness Rules

- same environment dynamics per scenario
- same observation channels where applicable
- same action bounds
- same evaluation horizon
- same initial conditions
- same disturbance schedules
- same uncertainty schedules per seed
- same total environment steps
- same evaluation frequency
- same model selection rule
- deterministic evaluation for final learned-policy benchmarking wherever applicable

## Main Result Framing

- The paper reproduction remains faithful to the original methods.
- `SAC`, `TD3`, `TQC`, and `PPO` provide broader standardized RL context.
- `DR-CIRL` is the only new method.
- The key question is whether `DR-CIRL` improves robustness under uncertainty relative to reproduced CIRL without materially degrading nominal performance.

## Improvement Claim Rule

Do **not** claim improvement unless:

- multi-seed results support it
- robustness improves relative to reproduced CIRL
- nominal tracking is not materially degraded

## Limitations To Acknowledge

- paper and official code contain a few implementation discrepancies
- final quantitative claims should come from CUDA runs, not local smoke tests
- optional baselines like `TQC` and optional Tier 2 scenarios should not distract from the core reproduction and `DR-CIRL` study
