# Summary For Student

## What This Project Is Doing

This project has two connected goals:

1. faithfully reproduce the CIRL paper on the nonlinear CSTR
2. add exactly one extension, `DR-CIRL`, to test whether domain randomization improves robustness

The paper-faithful methods stay unchanged:

- `Static PID`
- `Pure-RL (paper baseline)`
- `CIRL reproduced`
- `CIRL high-op extended (paper reproduction only)`

The only new method is:

- `DR-CIRL`

Modern RL baselines are added only for broader comparison:

- `SAC`
- `TD3`
- `TQC`
- `PPO`

## Why The Repo Starts With Notes And Configs

The paper and official code have a few inconsistencies, so we lock the source of truth first instead of jumping straight into implementation. That avoids building the wrong environment, reward, setpoint schedule, or controller interface.

The key files to read first are:

- `paper_notes.md`
- `reproduction_checklist.md`
- `configs/base.yaml`
- `configs/methods/*.yaml`
- `configs/scenarios/*.yaml`

## Main Technical Takeaways From The Official Code

- The plant is a nonlinear CSTR with states `C_A, C_B, C_C, T, V`.
- The learned controllers do not observe the full state directly.
- The main observation uses a 15-dimensional history/setpoint vector built from `C_B`, `T`, `V`, and setpoint history.
- The environment returns a positive tracking/control **cost**, even though official scripts print `-cost` as “reward”.
- CIRL outputs normalized PID parameters, not direct physical control moves.
- The paper baseline direct-action RL outputs normalized direct control actions.

## Planned Benchmarks

Tier 1:

- nominal tracking
- disturbance rejection
- high operating point
- parametric uncertainty

Tier 2, only if time and compute allow:

- measurement noise
- actuator saturation
- unseen setpoints

## How To Explain The Contribution

When you present the project, the correct framing is:

- the paper reproduction remains faithful to the original methods
- `SAC`, `TD3`, `TQC`, and `PPO` are added as standardized modern RL baselines
- `DR-CIRL` is the project’s only new method
- `DR-CIRL` changes only the training distribution through domain randomization

## What Counts As Success

The strongest acceptable claim is:

- `DR-CIRL` improves robustness under uncertainty relative to reproduced CIRL without materially degrading nominal tracking

It is **not** enough to claim success only because `DR-CIRL` beats the paper’s direct-action RL baseline.

## Current Status

At this stage the repo has:

- documentation scaffold
- experiment configs
- dependency specs
- setup script

Implementation of controllers, environments, training, evaluation, and plotting code is expected to follow on top of this locked scaffold.
