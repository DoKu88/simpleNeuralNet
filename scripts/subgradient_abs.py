"""
subgradient_abs.py

Subgradients of y = |x|:
  gradient = -1        for x < 0
  gradient = +1        for x > 0
  subgradient ∈ [-1,1] at x = 0  (the kink — any supporting line is valid)

Two panels:
  Left  — y = |x| with the fan of valid subgradient lines through the origin
  Right — graph of the subdifferential ∂|x| as a set-valued function
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

BG        = "#0e1117"
BLUE_LITE = "#93c5fd"
ORANGE    = "#f97316"
GREEN     = "#4ade80"
WHITE     = "#f8fafc"

LIM = 2.2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG)
fig.patch.set_facecolor(BG)
plt.subplots_adjust(wspace=0.38, left=0.07, right=0.97, top=0.88, bottom=0.13)

def _style(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#3a3a3a")
    ax.tick_params(colors=WHITE, labelsize=10)

_style(ax1)
_style(ax2)

# ═══════════════════════════════════════════════════════════════════════════════
# Left panel — y = |x| with subgradient fan
# ═══════════════════════════════════════════════════════════════════════════════
xv = np.linspace(-LIM, LIM, 800)
ax1.plot(xv, np.abs(xv), color=BLUE_LITE, lw=3.0, zorder=4, label=r"$y = |x|$")

# Interior subgradient lines: faint solid lines for slopes strictly between -1 and 1
x_fan = np.linspace(-LIM * 0.9, LIM * 0.9, 200)
for s in np.linspace(-1, 1, 11)[1:-1]:
    ax1.plot(x_fan, s * x_fan, color=ORANGE, lw=0.8, alpha=0.22, zorder=2)

# Boundary subgradients (slope = ±1) — bolder, in legend
ax1.plot(x_fan, -x_fan, color=ORANGE, lw=2.2, alpha=0.9, zorder=3,
         label=r"Subgradient Lines, Slope $g \in [-1,\,1]$")
ax1.plot(x_fan,  x_fan, color=ORANGE, lw=2.2, alpha=0.9, zorder=3)

# Kink point
ax1.scatter([0], [0], s=110, color=WHITE, edgecolors=BLUE_LITE,
            linewidths=2.0, zorder=10)

# Kink annotation
ax1.annotate(
    r"Any slope $g \in [-1,\,1]$" + "\n" + "valid subgradient",
    xy=(0, 0), xytext=(0.45, -0.22),
    fontsize=10.5, color=ORANGE, ha="left", va="top",
    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3, shrinkB=7),
)

# Faint region between boundary lines to highlight the fan
x_shade = np.linspace(0, LIM * 0.9, 200)
ax1.fill_between(x_shade,  x_shade, -x_shade, alpha=0.07, color=ORANGE, zorder=1)
x_shade_l = np.linspace(-LIM * 0.9, 0, 200)
ax1.fill_between(x_shade_l, x_shade_l, -x_shade_l, alpha=0.07, color=ORANGE, zorder=1)

ax1.axhline(0, color="#3a3a3a", lw=0.8)
ax1.axvline(0, color="#3a3a3a", lw=0.8)
ax1.set_xlim(-LIM, LIM)
ax1.set_ylim(-0.45, LIM)
ax1.set_xlabel("$x$", color=WHITE, fontsize=14)
ax1.set_ylabel("$y$", color=WHITE, fontsize=14, rotation=0, labelpad=14)
ax1.set_title(r"Subgradients at $x = 0$",
              color=WHITE, fontsize=13, fontweight="bold", pad=12)
ax1.legend(fontsize=11, facecolor="#1a1e2e", edgecolor="#555",
           labelcolor=WHITE, framealpha=0.85, loc="upper center")

# ═══════════════════════════════════════════════════════════════════════════════
# Right panel — subdifferential ∂|x| as a set-valued function
# ═══════════════════════════════════════════════════════════════════════════════
x_neg = np.linspace(-LIM, -0.02, 300)
x_pos = np.linspace( 0.02,  LIM, 300)

# Gradient lines for x ≠ 0
ax2.plot(x_neg, -np.ones_like(x_neg), color=ORANGE, lw=2.8)
ax2.plot(x_pos,  np.ones_like(x_pos),  color=ORANGE, lw=2.8)

# Open circles: gradient is not defined as a single value at x = 0
ax2.scatter([0, 0], [-1, 1], s=90, color=BG,
            edgecolors=ORANGE, linewidths=2.2, zorder=6)

# Vertical segment at x = 0: the full subgradient set [-1, 1]
ax2.plot([0, 0], [-1, 1], color=ORANGE, lw=3.5, zorder=5,
         solid_capstyle="round")
# Closed filled dots at the endpoints of the segment
ax2.scatter([0, 0], [-1, 1], s=90, color=ORANGE, zorder=8)

# ── Labels ────────────────────────────────────────────────────────────────────
ax2.text(-1.3, -1.28, r"$\partial|x| = \{-1\}$",
         color=ORANGE, fontsize=11, ha="center")
ax2.text( 1.3,  1.24, r"$\partial|x| = \{+1\}$",
         color=ORANGE, fontsize=11, ha="center")
# y-axis tick labels
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels([r"$-1$", r"$0$", r"$+1$"], color=WHITE, fontsize=11)

ax2.axhline(0, color="#3a3a3a", lw=0.8)
ax2.axvline(0, color="#3a3a3a", lw=0.8)
ax2.set_xlim(-LIM, LIM)
ax2.set_ylim(-1.75, 1.75)
ax2.set_xlabel("$x$", color=WHITE, fontsize=14)
ax2.set_ylabel(r"$\partial|x|$", color=WHITE, fontsize=14, rotation=0, labelpad=28)
ax2.set_title(r"Subdifferential  $\partial|x|$",
              color=WHITE, fontsize=13, fontweight="bold", pad=12)

# ── Save ──────────────────────────────────────────────────────────────────────
out = OUTPUT_DIR / f"{stamp}_subgradient_abs.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
plt.close(fig)
