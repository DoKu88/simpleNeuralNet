"""
lasso_diamond_classic.py

Classic 2-D L1 "diamond" diagram — dark aesthetic.
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
BLUE      = "#2563eb"
BLUE_LITE = "#93c5fd"

T   = 1.5
LIM = 2.5

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
ax.set_facecolor(BG)
for sp in ax.spines.values():
    sp.set_edgecolor("#3a3a3a")
ax.tick_params(colors="white", labelsize=10)

# ── Diamond ───────────────────────────────────────────────────────────────────
verts = np.array([[T, 0], [0, T], [-T, 0], [0, -T]])
# Fill and outline drawn separately so the edge alpha is independent of the fill
ax.add_patch(plt.Polygon(verts, closed=True,
                         facecolor=BLUE, alpha=0.35,
                         edgecolor="none", zorder=3))
ax.add_patch(plt.Polygon(verts, closed=True,
                         facecolor="none",
                         edgecolor="#ff0000", linewidth=2.5, zorder=6))

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.axhline(0, color="#3a3a3a", lw=0.8)
ax.axvline(0, color="#3a3a3a", lw=0.8)

# ── Styling ───────────────────────────────────────────────────────────────────
ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_aspect("equal")
ax.set_xlabel("$w_1$", color="white", fontsize=14)
ax.set_ylabel("$w_2$", color="white", fontsize=14, rotation=0, labelpad=16)
ax.set_title(r"$\|w\|_1 = |w_1| + |w_2| \leq \lambda$",
             color="white", fontsize=13, fontweight="bold", pad=12)

# ── Corner labels ─────────────────────────────────────────────────────────────
d = 0.12
ax.text( T + d,  0,      rf"$\lambda$",  color=BLUE_LITE, fontsize=11, va="center")
ax.text(-T - d,  0,     rf"$-\lambda$",  color=BLUE_LITE, fontsize=11, va="center", ha="right")
ax.text( 0,      T + d,  rf"$\lambda$",  color=BLUE_LITE, fontsize=11, ha="center")
ax.text( 0,     -T - d, rf"$-\lambda$",  color=BLUE_LITE, fontsize=11, ha="center", va="top")

plt.tight_layout()
out = OUTPUT_DIR / f"{stamp}_lasso_diamond_classic.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
plt.close(fig)
