"""Generate conceptual block diagrams for the report."""

from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from src.plotting.publication_style import apply_publication_style, finalize_figure, save_figure
from src.utils.paths import FIGURES_DIR


def _panel(ax, x: float, y: float, w: float, h: float, *, facecolor: str, edgecolor: str = "#222222", radius: float = 0.03):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    return patch


def _circle(ax, x: float, y: float, r: float = 0.028, *, facecolor: str = "white", edgecolor: str = "#222222", lw: float = 1.0):
    circ = Circle((x, y), r, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(circ)
    return circ


def _arrow(ax, start, end, *, text: str | None = None, text_xy=None, color: str = "#222222", lw: float = 1.2, style: str = "-|>"):
    arr = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, linewidth=lw, color=color)
    ax.add_patch(arr)
    if text:
        tx, ty = text_xy if text_xy is not None else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(tx, ty, text, fontsize=11, ha="center", va="center", color=color)
    return arr


def plot_rl_loop_diagram() -> tuple[Path, Path]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _panel(ax, 0.36, 0.58, 0.24, 0.26, facecolor="#f3f4f6")
    _panel(ax, 0.40, 0.12, 0.22, 0.28, facecolor="#f8fafc")
    ax.text(0.48, 0.79, "RL Agent", fontsize=14, ha="center")
    ax.text(0.51, 0.14, "Environment", fontsize=14, ha="center")

    # small network icon
    xs = [0.40, 0.45, 0.50, 0.55, 0.60]
    ys_layers = [
        [0.71, 0.64],
        [0.76, 0.70, 0.64, 0.58],
        [0.74, 0.67, 0.60],
        [0.76, 0.70, 0.64, 0.58],
        [0.71, 0.64],
    ]
    for i in range(len(xs) - 1):
        for y0 in ys_layers[i]:
            for y1 in ys_layers[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [y0, y1], color="#6b7280", linewidth=0.8)
    for x, ys in zip(xs, ys_layers):
        for y in ys:
            _circle(ax, x, y, r=0.010, facecolor="white", edgecolor="#555555", lw=0.9)

    # simple reactor sketch
    reactor = Rectangle((0.455, 0.19), 0.10, 0.12, facecolor="white", edgecolor="#444444", linewidth=1.0)
    ax.add_patch(reactor)
    ax.plot([0.455, 0.47, 0.47, 0.54, 0.54, 0.555], [0.31, 0.34, 0.20, 0.20, 0.34, 0.31], color="#444444", linewidth=1.0)
    ax.text(0.476, 0.205, "A", fontsize=10)
    ax.text(0.525, 0.205, "B", fontsize=10)
    _arrow(ax, (0.50, 0.25), (0.525, 0.25), color="#111827")
    _arrow(ax, (0.525, 0.25), (0.50, 0.25), color="#111827", style="<|-")

    _arrow(ax, (0.32, 0.74), (0.36, 0.74), text=r"State ($x_t$)", text_xy=(0.27, 0.81))
    _arrow(ax, (0.23, 0.64), (0.36, 0.64), text=r"Reward ($r_t$)", text_xy=(0.24, 0.70))
    _arrow(ax, (0.60, 0.67), (0.76, 0.67), text=r"Action" + "\n" + r"($u_t$)", text_xy=(0.80, 0.64))
    _arrow(ax, (0.76, 0.67), (0.76, 0.26), color="#222222", style="-")
    _arrow(ax, (0.76, 0.26), (0.62, 0.26), color="#222222")
    _arrow(ax, (0.40, 0.26), (0.23, 0.26), color="#222222", style="-")
    _arrow(ax, (0.23, 0.26), (0.23, 0.64), color="#222222", style="-")
    _arrow(ax, (0.23, 0.19), (0.23, 0.74), color="#222222", style="-")
    _arrow(ax, (0.23, 0.19), (0.36, 0.19), text=r"State ($x_{t+1}$)", text_xy=(0.19, 0.11))
    ax.text(0.31, 0.32, r"$r_{t+1}$", fontsize=11)

    finalize_figure(fig)
    return save_figure(fig, FIGURES_DIR / "paper_rl_loop_diagram")


def plot_policy_network_diagram() -> tuple[Path, Path]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = [0.12, 0.30, 0.48, 0.66, 0.84]
    layers = [
        [0.72, 0.50, 0.28],
        [0.80, 0.64, 0.48, 0.32, 0.16],
        [0.80, 0.64, 0.48, 0.32, 0.16],
        [0.80, 0.64, 0.48, 0.32, 0.16],
        [0.65, 0.43, 0.21],
    ]
    layer_titles = ["Input\nlayer", "Hidden\nlayer $h_1$", "Hidden\nlayer $h_2$", "Hidden\nlayer $h_n$", "Output\nlayer"]

    for i in range(len(x_positions) - 1):
        for y0 in layers[i]:
            for y1 in layers[i + 1]:
                ax.plot([x_positions[i], x_positions[i + 1]], [y0, y1], color="#9ca3af", linewidth=0.8)

    for x, ys, title in zip(x_positions, layers, layer_titles):
        for y in ys:
            face = "#a3a3a3" if x in (x_positions[0], x_positions[-1]) else "#d4d4d8"
            _circle(ax, x, y, r=0.022, facecolor=face, edgecolor="#888888", lw=0.8)
        ax.text(x, 0.93, title, ha="center", va="center", fontsize=13)

    ax.text(0.055, 0.72, r"$x_1$", fontsize=12)
    ax.text(0.055, 0.50, r"$x_2$", fontsize=12)
    ax.text(0.055, 0.28, r"$x_n$", fontsize=12)
    ax.text(0.065, 0.39, r"$\vdots$", fontsize=18)
    ax.text(0.875, 0.65, r"$u_1$", fontsize=12)
    ax.text(0.875, 0.43, r"$u_2$", fontsize=12)
    ax.text(0.875, 0.21, r"$u_n$", fontsize=12)
    ax.text(0.885, 0.32, r"$\vdots$", fontsize=18)
    ax.text(0.50, 0.05, r"Deep policy network $\pi_\theta$", ha="center", fontsize=18)

    finalize_figure(fig)
    return save_figure(fig, FIGURES_DIR / "paper_policy_network_diagram")


def plot_cirl_architecture_diagram() -> tuple[Path, Path]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _panel(ax, 0.05, 0.08, 0.58, 0.82, facecolor="#2f7f96")
    _panel(ax, 0.63, 0.08, 0.22, 0.82, facecolor="#8ec5d6")
    _panel(ax, 0.85, 0.08, 0.10, 0.82, facecolor="#b6d8e5")
    ax.text(0.34, 0.87, "Neural Network", fontsize=20, ha="center")
    ax.text(0.74, 0.87, "PID Controller", fontsize=18, ha="center")
    ax.text(0.90, 0.87, "Process", fontsize=18, ha="center")

    _circle(ax, 0.10, 0.50, r=0.028)
    ax.text(0.10, 0.50, r"$s_t$", fontsize=16, ha="center", va="center")

    xs = [0.27, 0.43]
    ys = [0.76, 0.61, 0.46, 0.31, 0.16]
    outputs = [0.68, 0.50, 0.32]
    for x in xs:
        for y in ys:
            _circle(ax, x, y, r=0.025)
    for i, x in enumerate(xs):
        prev_x = 0.10 if i == 0 else xs[i - 1]
        prev_ys = [0.50] if i == 0 else ys
        for y0 in prev_ys:
            for y1 in ys:
                ax.plot([prev_x, x], [y0, y1], color="#111827", linewidth=0.9)
    out_x = 0.59
    labels = [r"$K_{p,t}$", r"$\tau_{i,t}$", r"$\tau_{d,t}$"]
    for y, label in zip(outputs, labels):
        _circle(ax, out_x, y, r=0.025)
        ax.text(out_x, y, label, fontsize=12, ha="center", va="center")
        for y0 in ys:
            ax.plot([xs[-1], out_x], [y0, y], color="#111827", linewidth=0.9)

    err_box = Rectangle((0.27, 0.11), 0.18, 0.05, facecolor="white", edgecolor="#444444", linewidth=0.8)
    ax.add_patch(err_box)
    ax.text(0.36, 0.135, r"$e_t = x_t^{\ast} - x_t$", fontsize=13, ha="center", va="center")
    _circle(ax, 0.59, 0.16, r=0.025)
    ax.text(0.59, 0.16, r"$e_t$", fontsize=14, ha="center", va="center")
    _arrow(ax, (0.45, 0.135), (0.565, 0.16), color="#111827")

    pid_boxes_y = [0.63, 0.50, 0.37]
    pid_text = [r"$K_{p,t}(\Delta e_t)$", r"$K_{p,t}/\tau_{i,t}(e_t)$", r"$K_{p,t}\tau_{d,t}(\Delta^2 e_t)$"]
    for y, txt in zip(pid_boxes_y, pid_text):
        box = Rectangle((0.65, y - 0.05), 0.12, 0.10, facecolor="white", edgecolor="#4b5563", linewidth=0.8)
        ax.add_patch(box)
        ax.text(0.71, y, txt, fontsize=13, ha="center", va="center")
    sigma = Rectangle((0.79, 0.46), 0.04, 0.08, facecolor="white", edgecolor="#4b5563", linewidth=0.8)
    ax.add_patch(sigma)
    ax.text(0.81, 0.50, r"$\Sigma$", fontsize=16, ha="center", va="center")
    u_box = Rectangle((0.87, 0.46), 0.05, 0.08, facecolor="white", edgecolor="#4b5563", linewidth=0.8)
    ax.add_patch(u_box)
    ax.text(0.895, 0.50, r"$u_t$", fontsize=16, ha="center", va="center")

    for y in outputs:
        _arrow(ax, (0.615, y), (0.65, 0.63 if y > 0.6 else 0.50 if y > 0.4 else 0.37), color="#111827")
    _arrow(ax, (0.615, 0.16), (0.65, 0.50), color="#111827")
    for y in pid_boxes_y:
        _arrow(ax, (0.77, y), (0.79, 0.50), color="#111827")
    _arrow(ax, (0.83, 0.50), (0.87, 0.50), color="#111827")

    finalize_figure(fig)
    return save_figure(fig, FIGURES_DIR / "paper_cirl_architecture_diagram")


def main() -> None:
    outputs = [
        plot_rl_loop_diagram(),
        plot_policy_network_diagram(),
        plot_cirl_architecture_diagram(),
    ]
    for png_path, pdf_path in outputs:
        print(f"generated: {png_path}")
        print(f"generated: {pdf_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
