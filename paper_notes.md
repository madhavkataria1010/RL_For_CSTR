# Paper Notes: Control-Informed Reinforcement Learning for Chemical Processes

## Locked References

- Paper title: **Control-Informed Reinforcement Learning for Chemical Processes**
- arXiv: `2408.13566`
- journal DOI: `10.1021/acs.iecr.4c03233`
- official repository: `OptiMaL-PSE-Lab/CIRL`
- pinned reference commit: `c3fd22580a15d1e008570c08e78b38cdd887ef2c`
- commit message at pinned reference: `Implemented RGA`

## Problem Setup

The official repository models a nonlinear continuously stirred tank reactor with consecutive reactions:

- `A -> B`
- `B -> C`

The control task is setpoint tracking and disturbance rejection while regulating:

- product concentration `C_B`
- reactor volume `V`

Manipulated variables are:

- cooling jacket temperature `T_c`
- inlet flow rate `F_in`

## Process Model From Official Code

Official model file: `CIRL/cstr_model.py`

State vector used in the ODE:

- `C_A`
- `C_B`
- `C_C`
- `T`
- `V`

Nominal parameters in the code:

- `T_f = 350 K`
- `C_A,in = 1 mol/m^3`
- `F_out = 100`
- `rho = 1000`
- `C_p = 0.239`
- `UA = 5e4`
- `k0_AB = 7.2e10`
- `EoverR_AB = 8750`
- `k0_BC = 8.2e10`
- `EoverR_BC = 10750`
- reaction-rate form:
  - `rA = k0_AB * exp(-EoverR_AB / T) * C_A`
  - `rB = k0_BC * exp(-EoverR_BC / T) * C_B`

Initial steady-state values in the environment:

- `C_A,ss = 0.80`
- `T_ss = 327`
- `V_ss = 102`
- initial control history seeds:
  - `T_c = 300`
  - `F_in = 100`

Time grid in official code:

- `t = linspace(0, 100, ns)`

Paper-style plots in the official scripts use:

- `t = linspace(0, 25, ns)`

This mismatch must be documented as a paper-vs-code discrepancy rather than silently normalized away.

## Observation Space

The trainable policies do **not** receive the full ODE state directly.

Default RL/CIRL observation is a 15-dimensional vector made from:

- current measured outputs: `C_B(t), T(t), V(t)`
- one-step history: `C_B(t-1), T(t-1), V(t-1)`
- two-step history: `C_B(t-2), T(t-2), V(t-2)`
- current setpoints: `C_B,sp(t), V_sp(t)`
- one-step setpoint history: `C_B,sp(t-1), V_sp(t-1)`
- two-step setpoint history: `C_B,sp(t-2), V_sp(t-2)`

The official state index selection is:

`[1, 3, 4, 6, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20]`

Official observation normalization bounds:

- low: `[0, 350, 90, 0, 350, 90, 0, 350, 90, 0, 99, 0, 99, 0, 99]`
- high: `[1, 390, 102, 1, 390, 102, 1, 390, 102, 1, 101, 1, 101, 1, 101]`

Disturbance-observed variant:

- appends `C_A,in`
- total observation size becomes `16`

## Control Input And Action Space

Physical manipulated variables:

- `T_c`
- `F_in`

Paper baseline and modern direct-action RL interface:

- normalized action space: `[-1, 1]^2`
- physical mapping in `normRL=True` branch:
  - `T_c in [290, 400]`
  - `F_in in [99, 105]`

CIRL / PID-parameterized interface:

- normalized action space: `[-1, 1]^6`
- mapped using `x_norm`
- official normalized-to-physical PID parameter bounds:
  - lower: `[-5, 0, 0.01, 0, 0, 0.01]`
  - upper: `[25, 20, 10, 1, 2, 1]`

The six CIRL outputs correspond to:

- `Kp_cb`
- `taui_cb`
- `taud_cb`
- `Kp_v`
- `taui_v`
- `taud_v`

The embedded PID layer uses velocity-form updates and clamps:

- `T_c` to `[290, 450]`
- `F_in` to `[99, 105]`

Note the official code therefore uses different `T_c` upper bounds for:

- pure direct-action RL: `400`
- PID/CIRL action application: `450`

This discrepancy must stay documented.

## Reward / Cost Definition

The official environment returns a **positive stage cost** to be minimized, not a reward to be maximized.

With setpoint error vector `e = [C_B,sp - C_B, V_sp - V]`, the official cost is:

`e[0]^2 + e[1]^2 / 10 + 0.0005 * (Delta T_c)^2 + 0.005 * (Delta F_in)^2`

Important implications:

- the environment variable is called `rew`, but it is a cost
- official plotting scripts multiply totals by `-1` before printing them as “reward”
- modern RL baselines in this project must maximize `-cost` while logging both views explicitly

## Controller Architecture

Official trainable policy file: `CIRL/cirl_policy.py`

Network structure used for both direct-action RL and CIRL:

- fully connected layer 1
- activation: `ReLU`
- fully connected layer 2
- output layer
- final `tanh`

Nominal paper network sizes in official training code:

- `Pure-RL (paper baseline)`: hidden sizes `128, 128`, output size `2`
- `CIRL reproduced`: hidden sizes `16, 16`, output size `6`

Important code detail:

- `n_layers=1` is passed around
- an extra `ModuleList` of hidden layers is created
- those extra layers are **not used in the forward pass**

This is an implementation detail that must be preserved for faithful reproduction unless an explicitly documented reproduction fix is required later.

## Optimizer Details From Official Code

Official training file: `CIRL/training_CIRL.py`

Training loop settings:

- random search initializations: `30`
- parameter initialization range: uniform in `[-0.1, 0.1]`
- PSO optimizer: `ParticleSwarmOptimizer`
- PSO inertial weight: `0.6`
- PSO particle count: `30`
- PSO iterations: `150`
- criterion repeats per policy evaluation: `3`
- official training repetitions for figure generation: `10`

Static PID tuning from `CIRL/static_pid_gains.py`:

- optimizer: `scipy.optimize.differential_evolution`
- bounds: `(-1, 1)` for each of the six normalized gains
- `maxiter = 150`
- tuning evaluation repetitions: `1`
- tuning horizon uses `ns = 120 * 3`

## Official Setpoints And Scenarios

### Nominal Training

The non-test training environment cycles across three setpoint profiles:

- profile 1: `C_B,sp = [0.70, 0.75, 0.86]`
- profile 2: `C_B,sp = [0.10, 0.20, 0.30]`
- profile 3: `C_B,sp = [0.40, 0.50, 0.60]`
- `V_sp = 100` throughout

Each segment lasts `ns / 3` steps internally, and the environment chains three such profiles before episode termination in training mode.

### Nominal Test / Held-Out Setpoint Profile

When `test=True` and `highop=False`, the official setpoint profile becomes:

- `C_B,sp = [0.075, 0.45, 0.725]`
- `V_sp = 100`

This is the held-out nominal test profile used by the paper-style evaluation scripts.

### High-Operating-Point Scenario

Training profile when `highop=True`:

- `C_B,sp = [0.50, 0.90, 0.90]`

Test profile when `test=True` and `highop=True`:

- `C_B,sp = [0.45, 0.88, 0.88]`

### Disturbance Scenario

When `dist=True`:

- setpoint is fixed at `C_B,sp = 0.4`
- disturbance is applied after `i > 70`

Official disturbance values:

- training disturbance sequence: `C_A,in = [1.7, 1.6, 1.9]`
- test disturbance value: `C_A,in = 1.75`

The disturbance-observed variant exists in code but the published comparison emphasized the non-observed case.

## Baselines Used In The Paper

From the paper and official repo structure:

- `Static PID`
- direct-action RL baseline, referred to in this project as `Pure-RL (paper baseline)`
- `CIRL`
- high-operating-point extended CIRL training, treated here as reproduction-only

The paper baseline is **not** renamed to vanilla RL in this repo.

## Metrics And Paper-Style Quantities

Official scripts print mean negative cumulative cost as “reward”.

Project-level benchmark metrics later added on top of reproduction:

- IAE
- ISE
- ITAE
- RMSE
- overshoot
- settling time
- control variation
- constraint violations
- recovery time
- average rank

Paper-style scalar targets already locked for the reproduction summary:

- nominal test reward:
  - `Pure-RL (paper baseline): -2.08`
  - `CIRL reproduced: -1.33`
  - `Static PID: -1.77`
- high-operating-point reward:
  - `CIRL initial: -4.04`
  - `CIRL extended: -2.07`
  - `Static PID: -6.81`
- disturbance reward:
  - `Pure-RL (paper baseline): -1.76`
  - `CIRL reproduced: -1.38`

## Exact Figures / Tables To Reproduce

Official repo figure scripts and output files indicate these core reproduction targets:

1. network-size comparison figure
   - script: `CIRL/network_size_vis.py`
   - output: `plots/network_size_analysis.pdf`
2. learning-curve figure over 10 repetitions
   - script: `CIRL/learning_curve_vis.py`
   - output: `plots/lc_sp_newobs_0306.pdf`
3. nominal setpoint tracking comparison
   - script: `CIRL/SP_vis_paper.py`
   - outputs:
     - `plots/RLvsRLPID_states.pdf`
     - `plots/RLvsRLPIDsp_PID.pdf`
4. high-operating-point comparison
   - script: `CIRL/SP_vis_paper_highop.py`
   - outputs:
     - `plots/RLvsRLPID_states_highop.pdf`
     - `plots/RLvsRLPIDsp_PID_highop.pdf`
5. disturbance-response comparison
   - script: `CIRL/dist_vis_presentation.py`
   - output: `plots/disturbance_presentation.pdf`
6. static PID gain artifacts
   - `data/constant_gains.npy`
   - `data/constant_gains_highop.npy`
7. RGA appendix-style analysis
   - script: `CIRL/RGA.py`

## Artifact Replay Notes

The official nominal plotting script uses saved `.pkl` artifacts rather than only final `.pth` files:

- nominal RL example:
  - `results_rl_network_rep_newobs_0.pkl`
  - selects `inter[1]["p_list"][149]`
- nominal CIRL example:
  - `results_pid_network_rep_newobs_1.pkl`
  - selects `inter[0]["p_list"][149]`

High-op plotting script compares:

- high-op trained CIRL from `best_policy_pid_highop_0.pth`
- low-op-trained CIRL loaded from `results_pid_network_rep_newobs_1.pkl`
- static high-op PID gains from `constant_gains_highop.npy`

## Paper-Vs-Code Discrepancies To Preserve

1. The environment integrates over `0..100`, but official paper-style plots use `0..25`.
2. The environment variable named `rew` is a cost, while official scripts report `-cost` as reward.
3. The direct-action RL branch maps `T_c` to `[290, 400]`, while PID application clamps `T_c` to `[290, 450]`.
4. The official disturbance training values are `[1.7, 1.6, 1.9]`; earlier paper text references may differ.
5. `n_layers` is exposed in policy constructors, but the additional stored layers are not used in the forward pass.
6. Static PID tuning is performed on a `test=True` environment with `ns = 360`, which differs from the training-loop environment construction.

These discrepancies are part of the locked reproduction context and should be documented, not silently “fixed”.
