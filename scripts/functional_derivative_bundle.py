"""
functional_derivative_bundle.py

Visualizes the family of curves  y_ε(x) = y(x) + ε·η(x)
for a sweep of ε values, illustrating the concept of variation
in function space and the functional derivative.

Design (v3)
───────────
• All floating annotations live in the LEFT half of the plot (x < π),
  where the curves spread out cleanly.
• The function-space inset sits in the UPPER-RIGHT corner, above the
  convergence region — no overlap with labelled curves.
• The axis title is replaced by a short fig-level subtitle.
• Vertical displacement guides are on the RIGHT half (π < x < 5).
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import petname

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Aesthetics
# ---------------------------------------------------------------------------
BG_DARK    = "#1E1E1E"
BG_MID     = "#2A2A2A"
TEXT_COLOR = "#DDDDDD"
WHITE      = "white"
ETA_COLOR  = "#4C9BE8"

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def y_base(x: np.ndarray) -> np.ndarray:
    return np.sin(x) + 0.3 * np.sin(3 * x)


def eta(x: np.ndarray) -> np.ndarray:
    """η(0) = η(2π) = 0  (fixed-boundary condition)."""
    window = np.sin(x / 2) ** 2          # 0 → 1 → 0 over [0, 2π]
    return window * np.cos(2 * x - np.pi / 3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def style_ax(ax, ylabel: str, title: str | None = None) -> None:
    ax.set_facecolor(BG_MID)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.set_xlabel("x", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=11)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda v, _: {0.0: "0", round(np.pi, 5): "π",
                          round(2 * np.pi, 5): "2π"}.get(round(v, 5), f"{v:.1f}")
        )
    )
    ax.set_xticks([0, np.pi, 2 * np.pi])
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=10, pad=5)


def eps_alpha(eps: float) -> float:
    return max(0.15, 1.0 - 0.82 * abs(eps))


def eps_lw(eps: float) -> float:
    return max(0.7, 2.8 - 2.0 * abs(eps))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    stamp   = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    name    = petname.generate(2, "_")
    run_tag = f"{stamp}_{name}"

    x        = np.linspace(0, 2 * np.pi, 800)
    epsilons = np.linspace(-1.0, 1.0, 17)

    y0   = y_base(x)
    eta0 = eta(x)

    # ------------------------------------------------------------------ layout
    fig = plt.figure(figsize=(14, 9.8), facecolor=BG_DARK)

    # Two-line header: bold title + lighter subtitle
    fig.text(0.5, 0.988,
             r"Function-space bundle:  $y_\varepsilon(x) = y(x) + \varepsilon\,\eta(x)$",
             ha="center", va="top", fontsize=16, fontweight="bold", color=WHITE)
    fig.text(0.5, 0.953,
             r"Curves fade as $|\varepsilon|$ grows — emphasising the linearisation near $\varepsilon = 0$",
             ha="center", va="top", fontsize=10, color="#AAAAAA")

    gs = fig.add_gridspec(
        2, 1,
        left=0.07, right=0.83,
        top=0.930, bottom=0.07,
        hspace=0.44,
        height_ratios=[3, 1],
    )
    ax_bundle = fig.add_subplot(gs[0])
    ax_eta    = fig.add_subplot(gs[1])
    cbar_ax   = fig.add_axes([0.86, 0.07, 0.025, 0.860])

    style_ax(ax_bundle, r"$y_\varepsilon(x)$")
    style_ax(ax_eta, r"$\eta(x)$")
    ax_eta.set_title(
        r"Perturbation direction $\eta(x)$ — fixed endpoints: $\eta(0) = \eta(2\pi) = 0$",
        color=TEXT_COLOR, fontsize=10, pad=16,
    )


    # ------------------------------------------------------------------ colormap
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # ------------------------------------------------------------------ bundle curves
    for eps in epsilons:
        y_eps   = y0 + eps * eta0
        color   = cmap(norm(eps))
        is_zero = abs(eps) < 1e-9

        ax_bundle.plot(
            x, y_eps,
            color=color,
            linewidth=eps_lw(eps),
            alpha=eps_alpha(eps),
            zorder=5 if is_zero else 2,
            solid_capstyle="round",
        )

    # Bold ε = 0 curve drawn on top
    ax_bundle.plot(x, y0, color=WHITE, linewidth=2.8, zorder=6, solid_capstyle="round")
    ax_bundle.axhline(0, color="#555555", linewidth=0.6, zorder=1)

    # ------------------------------------------------------------------ η panel
    ax_eta.fill_between(x, eta0, alpha=0.20, color=ETA_COLOR)
    ax_eta.plot(x, eta0, color=ETA_COLOR, linewidth=2.0)
    ax_eta.axhline(0, color="#555555", linewidth=0.6, zorder=1)

    # [4] Fixed-endpoint markers
    for xv in (0.0, 2 * np.pi):
        ax_eta.plot(xv, 0.0, "o", markersize=7, zorder=10,
                    markerfacecolor=BG_MID, markeredgecolor=WHITE, markeredgewidth=1.5)
    ax_eta.text(0, -0.18, r"$\eta(0)=0$",
                color=TEXT_COLOR, fontsize=8.5, ha="center")
    ax_eta.text(2 * np.pi, -0.18, r"$\eta(2\pi)=0$",
                color=TEXT_COLOR, fontsize=8.5, ha="center")

    # ------------------------------------------------------------------ colorbar
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("ε", color=TEXT_COLOR, fontsize=14, rotation=0, labelpad=10)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=9)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    cb.outline.set_edgecolor("#444444")

    # ------------------------------------------------------------------ footnote
    fig.text(
        0.07, 0.006,
        r"$\eta(x)$ is the tangent direction in function space.  "
        r"$\varepsilon$ parameterises how far you step along it.  "
        r"The functional derivative $\delta F/\delta y$ is the rate of change "
        r"of $F[y_\varepsilon]$ as $\varepsilon \to 0$.",
        color="#888888", fontsize=8, ha="left", va="bottom",
    )

    # ------------------------------------------------------------------ save
    out_path = OUTPUT_DIR / f"{run_tag}_functional_derivative_bundle.png"
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
