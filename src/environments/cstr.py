"""Paper-faithful CSTR environment with clean configuration hooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from dataclasses import fields
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.integrate import odeint

from .perturbations import apply_rate_limit, sample_domain_randomization

try:  # pragma: no cover - optional dependency for SB3 compatibility
    from gymnasium import Env as GymEnv
    from gymnasium.spaces import Box as GymBox
except ModuleNotFoundError:  # pragma: no cover
    GymEnv = object
    GymBox = None


@dataclass
class BoxSpace:
    """Minimal Box-like space for projects not yet wired to Gym."""

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=float)
        self.high = np.asarray(self.high, dtype=float)
        self.shape = self.low.shape


def make_box_space(low: np.ndarray, high: np.ndarray):
    if GymBox is not None:
        return GymBox(low=np.asarray(low, dtype=np.float32), high=np.asarray(high, dtype=np.float32), dtype=np.float32)
    return BoxSpace(low=np.asarray(low, dtype=float), high=np.asarray(high, dtype=float))


PID_GAIN_LOW = np.array([-5.0, 0.0, 0.01, 0.0, 0.0, 0.01], dtype=float)
PID_GAIN_HIGH = np.array([25.0, 20.0, 10.0, 1.0, 2.0, 1.0], dtype=float)
OBS_LOW = np.array(
    [0, 350, 90, 0, 350, 90, 0, 350, 90, 0, 99, 0, 99, 0, 99],
    dtype=float,
)
OBS_HIGH = np.array(
    [1, 390, 102, 1, 390, 102, 1, 390, 102, 1, 101, 1, 101, 1, 101],
    dtype=float,
)
DIST_OBS_LOW = np.array([0, 350, 90, 0, 350, 90, 0, 99, 1], dtype=float)
DIST_OBS_HIGH = np.array([1, 390, 102, 1, 390, 103, 1, 101, 2], dtype=float)
PID_ACTION_LOW = np.full(6, -1.0, dtype=float)
PID_ACTION_HIGH = np.full(6, 1.0, dtype=float)
RL_ACTION_LOW = np.full(2, -1.0, dtype=float)
RL_ACTION_HIGH = np.full(2, 1.0, dtype=float)
PHYSICAL_PID_U_MIN = np.array([290.0, 99.0], dtype=float)
PHYSICAL_PID_U_MAX = np.array([450.0, 105.0], dtype=float)
PHYSICAL_RL_U_MIN = np.array([290.0, 99.0], dtype=float)
PHYSICAL_RL_U_MAX = np.array([400.0, 105.0], dtype=float)
OBSERVATION_INDICES = [1, 3, 4, 6, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20]


@dataclass
class CSTRProcessParameters:
    """Nominal nonlinear CSTR parameters from the official implementation."""

    tf: float = 350.0
    caf: float = 1.0
    fout: float = 100.0
    rho: float = 1000.0
    cp: float = 0.239
    ua: float = 5.0e4
    mdelh_ab: float = 5.0e3
    eoverr_ab: float = 8750.0
    k0_ab: float = 7.2e10
    mdelh_bc: float = 4.0e3
    eoverr_bc: float = 10750.0
    k0_bc: float = 8.2e10

    def apply_scales(
        self,
        caf_scale: float = 1.0,
        ua_scale: float = 1.0,
        k0_scale: float = 1.0,
    ) -> "CSTRProcessParameters":
        return replace(
            self,
            caf=self.caf * caf_scale,
            ua=self.ua * ua_scale,
            k0_ab=self.k0_ab * k0_scale,
            k0_bc=self.k0_bc * k0_scale,
        )


@dataclass
class MeasurementNoiseConfig:
    enabled: bool = True
    ca: float = 0.001
    cb: float = 0.001
    cc: float = 0.001
    t: float = 0.1
    v: float = 0.01


@dataclass
class DomainRandomizationConfig:
    enabled: bool = False
    ranges: Dict[str, float] = field(
        default_factory=lambda: {
            "caf_scale": 0.10,
            "ua_scale": 0.10,
            "k0_scale": 0.10,
        }
    )


@dataclass
class UncertaintyConfig:
    caf_scale: float = 1.0
    ua_scale: float = 1.0
    k0_scale: float = 1.0


@dataclass
class CSTRConfig:
    """Configurable environment settings with paper-faithful defaults."""

    ns: int = 120
    test: bool = False
    ds_mode: bool = False
    norm_rl: bool = False
    dist: bool = False
    dist_train: bool = False
    dist_obs: bool = False
    highop: bool = False
    paper_exact: bool = True
    seed: int | None = None
    time_start: float = 0.0
    time_end: float = 100.0
    ca_ss: float = 0.80
    t_ss: float = 327.0
    v_ss: float = 102.0
    initial_pid_action: tuple[float, float] = (302.0, 99.0)
    direct_action_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (290.0, 99.0),
        (400.0, 105.0),
    )
    pid_action_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (290.0, 99.0),
        (450.0, 105.0),
    )
    disturbance_start_step: int = 70
    training_disturbances: tuple[float, float, float] = (1.7, 1.6, 1.9)
    evaluation_disturbance: float = 1.75
    rate_limit: tuple[float, float] | None = None
    measurement_noise: MeasurementNoiseConfig = field(
        default_factory=MeasurementNoiseConfig
    )
    process: CSTRProcessParameters = field(default_factory=CSTRProcessParameters)
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> "CSTRConfig":
        payload: Dict[str, Any] = dict(data or {})
        payload.update(overrides)
        allowed = {field.name for field in fields(cls)}
        payload = {key: value for key, value in payload.items() if key in allowed}
        if "measurement_noise" in payload and not isinstance(
            payload["measurement_noise"], MeasurementNoiseConfig
        ):
            payload["measurement_noise"] = MeasurementNoiseConfig(
                **payload["measurement_noise"]
            )
        if "process" in payload and not isinstance(
            payload["process"], CSTRProcessParameters
        ):
            payload["process"] = CSTRProcessParameters(**payload["process"])
        if "domain_randomization" in payload and not isinstance(
            payload["domain_randomization"], DomainRandomizationConfig
        ):
            payload["domain_randomization"] = DomainRandomizationConfig(
                **payload["domain_randomization"]
            )
        if "uncertainty" in payload and not isinstance(
            payload["uncertainty"], UncertaintyConfig
        ):
            payload["uncertainty"] = UncertaintyConfig(**payload["uncertainty"])
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_profile(ns: int, values: Sequence[float]) -> np.ndarray:
    """Split a horizon into three near-equal segments."""

    counts = [ns // 3, ns // 3, ns - 2 * (ns // 3)]
    profile = []
    for value, count in zip(values, counts):
        profile.extend([float(value)] * int(count))
    return np.asarray(profile, dtype=float)


def build_paper_setpoint_profiles(
    ns: int,
    *,
    test: bool = False,
    dist: bool = False,
    highop: bool = False,
) -> list[np.ndarray]:
    """Replicate the setpoint schedules from the official environment."""

    if dist:
        return [np.full(ns, 0.4, dtype=float)] * 3
    if test and highop:
        return [build_profile(ns, (0.45, 0.88, 0.88))]
    if test:
        return [build_profile(ns, (0.075, 0.45, 0.725))]
    if highop:
        return [
            build_profile(ns, (0.5, 0.9, 0.9)),
            build_profile(ns, (0.1, 0.2, 0.3)),
            build_profile(ns, (0.4, 0.5, 0.6)),
        ]
    return [
        build_profile(ns, (0.70, 0.75, 0.86)),
        build_profile(ns, (0.1, 0.2, 0.3)),
        build_profile(ns, (0.4, 0.5, 0.6)),
    ]


def denormalize_pid_gains(action: np.ndarray) -> np.ndarray:
    """Map PID actions from ``[-1, 1]`` to physical gain ranges."""

    action = np.asarray(action, dtype=float)
    return ((action + 1.0) / 2.0) * (PID_GAIN_HIGH - PID_GAIN_LOW) + PID_GAIN_LOW


def denormalize_direct_action(action: np.ndarray) -> np.ndarray:
    """Map direct RL actions from ``[-1, 1]`` to physical actuator values."""

    action = np.asarray(action, dtype=float)
    return ((action + 1.0) / 2.0) * (
        PHYSICAL_RL_U_MAX - PHYSICAL_RL_U_MIN
    ) + PHYSICAL_RL_U_MIN


def pid_velocity_update(
    gains: np.ndarray,
    error: np.ndarray,
    error_history: np.ndarray,
    action_history: list[np.ndarray],
    time_window: np.ndarray,
) -> np.ndarray:
    """Official dynamic velocity-form PID update."""

    dt = float(time_window[1] - time_window[0])
    kp_cb, ti_cb, td_cb, kp_v, ti_v, td_v = np.asarray(gains, dtype=float)
    ti_cb = ti_cb + 1e-6
    ti_v = ti_v + 1e-6

    tc = (
        action_history[-1][0]
        + kp_cb * (error[0] - error_history[-1, 0])
        + (kp_cb / ti_cb) * error[0] * dt
        - kp_cb * td_cb * (error[0] - 2 * error_history[-1, 0] + error_history[-2, 0]) / dt
    )
    fin = (
        action_history[-1][1]
        + kp_v * (error[1] - error_history[-1, 1])
        + (kp_v / ti_v) * error[1] * dt
        - kp_v * td_v * (error[1] - 2 * error_history[-1, 1] + error_history[-2, 1]) / dt
    )
    return np.array(
        [
            np.clip(tc, PHYSICAL_PID_U_MIN[0], PHYSICAL_PID_U_MAX[0]),
            np.clip(fin, PHYSICAL_PID_U_MIN[1], PHYSICAL_PID_U_MAX[1]),
        ],
        dtype=float,
    )


def cstr_ode(
    x: np.ndarray,
    t: float,
    u: np.ndarray,
    params: CSTRProcessParameters,
) -> np.ndarray:
    """Nonlinear CSTR ODE system from the official implementation."""

    tc, fin = np.asarray(u, dtype=float)
    ca, cb, cc, temp, volume = np.asarray(x, dtype=float)

    r_a = params.k0_ab * np.exp(-params.eoverr_ab / temp) * ca
    r_b = params.k0_bc * np.exp(-params.eoverr_bc / temp) * cb

    dca_dt = (fin * params.caf - params.fout * ca) / volume - r_a
    dcb_dt = r_a - r_b - params.fout * cb / volume
    dcc_dt = r_b - params.fout * cc / volume
    dt_dt = (
        fin / volume * (params.tf - temp)
        + params.mdelh_ab / (params.rho * params.cp) * r_a
        + params.mdelh_bc / (params.rho * params.cp) * r_b
        + params.ua / volume / params.rho / params.cp * (tc - temp)
    )
    dv_dt = fin - params.fout
    return np.array([dca_dt, dcb_dt, dcc_dt, dt_dt, dv_dt], dtype=float)


class CSTRReactorEnv(GymEnv):
    """Source-of-truth-aligned environment for the paper-faithful experiments."""

    def __init__(
        self,
        config: CSTRConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        self.config = (
            config
            if isinstance(config, CSTRConfig)
            else CSTRConfig.from_mapping(config, **overrides)
        )
        self.rng = np.random.default_rng(self.config.seed)
        self.time_grid = np.linspace(
            self.config.time_start, self.config.time_end, self.config.ns
        )
        self.cb_profiles = build_paper_setpoint_profiles(
            self.config.ns,
            test=self.config.test,
            dist=self.config.dist,
            highop=self.config.highop,
        )
        self.v_profile = np.full(self.config.ns, 100.0, dtype=float)

        obs_low = DIST_OBS_LOW if self.config.dist_obs else OBS_LOW
        obs_high = DIST_OBS_HIGH if self.config.dist_obs else OBS_HIGH
        self.observation_space = make_box_space(low=obs_low, high=obs_high)
        if self.config.norm_rl:
            self.action_space = make_box_space(low=RL_ACTION_LOW, high=RL_ACTION_HIGH)
        elif self.config.ds_mode:
            self.action_space = make_box_space(
                low=PHYSICAL_RL_U_MIN,
                high=PHYSICAL_RL_U_MAX,
            )
        else:
            self.action_space = make_box_space(low=PID_ACTION_LOW, high=PID_ACTION_HIGH)

        self.full_state = np.zeros(21, dtype=float)
        self.raw_observation = np.zeros_like(self.observation_space.low)
        self.normalized_observation = np.zeros_like(self.observation_space.low)
        self.info: Dict[str, Any] = {}
        self.done = False
        self.i = 0
        self.sp_i = 0
        self.current_params = self.config.process
        self.current_caf = self.current_params.caf
        self.ts = np.array([0.0, 0.0], dtype=float)
        self.u_history: list[np.ndarray] = []
        self.e_history: list[np.ndarray] = []
        self.s_history: list[np.ndarray] = []

    def _sample_process_parameters(self) -> CSTRProcessParameters:
        params = self.config.process.apply_scales(
            caf_scale=self.config.uncertainty.caf_scale,
            ua_scale=self.config.uncertainty.ua_scale,
            k0_scale=self.config.uncertainty.k0_scale,
        )
        if self.config.domain_randomization.enabled:
            scales = sample_domain_randomization(
                self.rng, self.config.domain_randomization.ranges
            )
            params = params.apply_scales(**scales)
        return params

    def _build_initial_state(self, cb_des: float, v_des: float) -> np.ndarray:
        ca_ss, t_ss, v_ss = self.config.ca_ss, self.config.t_ss, self.config.v_ss
        return np.array(
            [
                ca_ss,
                0.0,
                0.0,
                t_ss,
                v_ss,
                ca_ss,
                0.0,
                0.0,
                t_ss,
                v_ss,
                ca_ss,
                0.0,
                0.0,
                t_ss,
                v_ss,
                cb_des,
                v_des,
                cb_des,
                v_des,
                cb_des,
                v_des,
            ],
            dtype=float,
        )

    def _current_setpoint(self) -> tuple[float, float]:
        cb_profile = self.cb_profiles[self.sp_i]
        return float(cb_profile[self.i]), float(self.v_profile[self.i])

    def _extract_observation(self) -> np.ndarray:
        obs = self.full_state[OBSERVATION_INDICES].astype(float)
        if self.config.dist_obs:
            obs = np.array(
                [
                    self.full_state[1],
                    self.full_state[3],
                    self.full_state[4],
                    self.full_state[6],
                    self.full_state[8],
                    self.full_state[9],
                    self.full_state[15],
                    self.full_state[16],
                    self.current_caf,
                ],
                dtype=float,
            )
        self.raw_observation = obs
        self.normalized_observation = (obs - self.observation_space.low) / (
            self.observation_space.high - self.observation_space.low
        )
        return self.normalized_observation.copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Reset back to the initial state of the current scenario."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "config_overrides" in options:
            self.config = CSTRConfig.from_mapping(
                self.config.to_dict(), **dict(options["config_overrides"])
            )

        self.i = 0
        self.sp_i = 0
        self.done = False
        self.current_params = self._sample_process_parameters()
        self.current_caf = self.current_params.caf
        cb_des, v_des = self._current_setpoint()
        self.full_state = self._build_initial_state(cb_des, v_des)
        self.u_history = []
        self.e_history = []
        self.s_history = []
        self.ts = self.time_grid[:2].astype(float)
        self.info = {
            "episode_index": self.sp_i,
            "step_index": self.i,
            "process_parameters": asdict(self.current_params),
            "disturbance_start_step": self.config.disturbance_start_step,
            "action_lower_bound": (
                PHYSICAL_RL_U_MIN.tolist() if self.config.norm_rl else PHYSICAL_PID_U_MIN.tolist()
            ),
            "action_upper_bound": (
                PHYSICAL_RL_U_MAX.tolist() if self.config.norm_rl else PHYSICAL_PID_U_MAX.tolist()
            ),
        }
        return self._extract_observation(), self.info.copy()

    def step(
        self,
        action_policy: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the reactor by one step."""

        if self.done:
            raise RuntimeError("Cannot call step() on a terminated episode.")

        cb_des, v_des = self._current_setpoint()
        self.full_state, reward = self._reactor(
            self.full_state,
            np.asarray(action_policy, dtype=float),
            cb_des,
            v_des,
        )

        self.i += 1
        if self.i == self.config.ns:
            if self.config.test or self.sp_i >= len(self.cb_profiles) - 1:
                self.done = True
            else:
                self.sp_i += 1
                self.i = 0
                self.current_caf = self.current_params.caf
                cb_des, v_des = self._current_setpoint()
                self.full_state = self._build_initial_state(cb_des, v_des)
                self.u_history = []
                self.e_history = []
                self.s_history = []

        if not self.done:
            step_idx = min(self.i + 1, len(self.time_grid) - 1)
            prev_idx = min(self.i, len(self.time_grid) - 2)
            self.ts = self.time_grid[[prev_idx, step_idx]].astype(float)

        observation = self._extract_observation()
        self.info.update(
            {
                "episode_index": self.sp_i,
                "step_index": self.i,
                "cb_setpoint": cb_des,
                "volume_setpoint": v_des,
                "reference": np.array([cb_des, v_des], dtype=float),
                "time": float(self.time_grid[min(self.i, len(self.time_grid) - 1)]),
                "cost": float(reward),
                "caf": self.current_caf,
                "terminated": self.done,
            }
        )
        return observation, float(reward), self.done, False, self.info.copy()

    def _resolve_action(
        self,
        action: np.ndarray,
        error: np.ndarray,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if self.config.ds_mode:
            u = np.asarray(action, dtype=float)
            info = {"u": u.copy()}
            return u, info

        if self.config.norm_rl:
            if self.i < 2:
                u = np.array(self.config.initial_pid_action, dtype=float)
            else:
                u = denormalize_direct_action(action)
            info = {"u": u.copy()}
            return u, info

        gains = denormalize_pid_gains(action)
        if self.i < 2:
            u = np.array(self.config.initial_pid_action, dtype=float)
        else:
            u = pid_velocity_update(
                gains,
                error,
                np.asarray(self.e_history, dtype=float),
                self.u_history,
                self.ts,
            )
        info = {"Ks": gains.copy(), "u": u.copy()}
        return u, info

    def _process_parameters_for_step(self) -> CSTRProcessParameters:
        if self.config.dist and self.i > self.config.disturbance_start_step:
            if self.config.dist_train:
                self.current_caf = self.config.training_disturbances[self.sp_i]
            else:
                self.current_caf = self.config.evaluation_disturbance
        else:
            self.current_caf = self.current_params.caf
        return replace(self.current_params, caf=self.current_caf)

    def _apply_measurement_noise(self, y_next: np.ndarray) -> np.ndarray:
        if not self.config.measurement_noise.enabled:
            return np.asarray(y_next, dtype=float)
        noise = np.array(
            [
                self.rng.uniform(-self.config.measurement_noise.ca, self.config.measurement_noise.ca),
                self.rng.uniform(-self.config.measurement_noise.cb, self.config.measurement_noise.cb),
                self.rng.uniform(-self.config.measurement_noise.cc, self.config.measurement_noise.cc),
                self.rng.uniform(-self.config.measurement_noise.t, self.config.measurement_noise.t),
                self.rng.uniform(-self.config.measurement_noise.v, self.config.measurement_noise.v),
            ],
            dtype=float,
        )
        return np.asarray(y_next, dtype=float) + noise

    def _build_next_full_state(
        self,
        current_state: np.ndarray,
        y_next: np.ndarray,
        cb_des: float,
        v_des: float,
    ) -> np.ndarray:
        current_process = current_state[:5]
        state_plus = np.zeros(21, dtype=float)
        state_plus[:5] = y_next
        state_plus[5:10] = current_process
        if self.i < 2:
            state_plus[10:15] = current_process
            state_plus[15:] = [cb_des, v_des, cb_des, v_des, cb_des, v_des]
        else:
            state_plus[10:15] = self.s_history[-1][:5]
            state_plus[15:17] = [cb_des, v_des]
            state_plus[17:19] = self.s_history[-1][15:17]
            state_plus[19:21] = self.s_history[-2][15:17]
        return state_plus

    def _reactor(
        self,
        state: np.ndarray,
        action: np.ndarray,
        cb_des: float,
        v_des: float,
    ) -> tuple[np.ndarray, float]:
        error = np.array([cb_des - state[1], v_des - state[4]], dtype=float)
        u, info = self._resolve_action(action, error)
        if self.u_history:
            previous_u = self.u_history[-1]
        else:
            previous_u = np.array(self.config.initial_pid_action, dtype=float)
        rate_limit = None
        if self.config.rate_limit is not None:
            rate_limit = np.asarray(self.config.rate_limit, dtype=float)
            u = apply_rate_limit(previous_u, u, rate_limit)

        if self.config.norm_rl:
            u = np.clip(u, PHYSICAL_RL_U_MIN, PHYSICAL_RL_U_MAX)
        else:
            u = np.clip(u, PHYSICAL_PID_U_MIN, PHYSICAL_PID_U_MAX)

        process_params = self._process_parameters_for_step()
        y = odeint(cstr_ode, state[:5], self.ts, args=(u, process_params))
        y_next = self._apply_measurement_noise(y[-1])
        state_plus = self._build_next_full_state(state, y_next, cb_des, v_des)

        if self.i == 0:
            u_change = np.zeros(2, dtype=float)
        else:
            u_change = (u - self.u_history[-1]) ** 2

        self.e_history.append(error.copy())
        self.u_history.append(u.copy())
        self.s_history.append(state.copy())
        self.info = info

        cost = error[0] ** 2
        cost += error[1] ** 2 / 10.0
        cost += 0.0005 * u_change[0] + 0.005 * u_change[1]
        return state_plus, float(cost)
