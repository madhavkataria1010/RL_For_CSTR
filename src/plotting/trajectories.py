"""Trajectory plotting helpers."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from .publication_style import apply_publication_style, finalize_figure


def plot_rollout_trajectory(
    *,
    time: np.ndarray,
    output: np.ndarray,
    reference: np.ndarray | None = None,
    action: np.ndarray | None = None,
    output_label: str = "Output",
    output_unit: str = "",
    action_label: str = "Control input",
    action_unit: str = "",
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    apply_publication_style()
    has_action = action is not None
    fig, axes = plt.subplots(2 if has_action else 1, 1, sharex=True)
    axes_arr = np.atleast_1d(axes)

    axes_arr[0].plot(time, output, label=output_label, linewidth=2.0)
    if reference is not None:
        axes_arr[0].plot(time, reference, label="Reference", linestyle="--", linewidth=1.6)
    ylabel = output_label if not output_unit else f"{output_label} [{output_unit}]"
    axes_arr[0].set_ylabel(ylabel)
    axes_arr[0].legend(loc="best")

    if has_action:
        action_arr = np.asarray(action, dtype=float)
        if action_arr.ndim == 1:
            axes_arr[1].plot(time, action_arr, label=action_label, linewidth=2.0)
        else:
            for column in range(action_arr.shape[1]):
                axes_arr[1].plot(time, action_arr[:, column], label=f"{action_label} {column}", linewidth=2.0)
            axes_arr[1].legend(loc="best")
        action_ylabel = action_label if not action_unit else f"{action_label} [{action_unit}]"
        axes_arr[1].set_ylabel(action_ylabel)
        axes_arr[1].set_xlabel("Time")
    else:
        axes_arr[0].set_xlabel("Time")

    return finalize_figure(fig, title=title), axes_arr
