"""
lasso_kkt_conditions.py

KKT subgradient conditions visualised directly on the 2-D L1 diamond.

Each edge of the diamond is a region of (w1, w2) space with both weights
nonzero — the residual correlations r_i = x_i^T(y-Xw) are pinned to ±λ.
The four corners are the sparse solutions where one weight = 0 and the
corresponding correlation is free inside [-λ, λ] (Case 3).
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#0e1117"
BLUE      = "#2563eb"
BLUE_LITE = "#93c5fd"
C_NE      = "#00d4aa"   # w1>0, w2>0
C_NW      = "#c084fc"   # w1<0, w2>0
C_SW      = "#ff6b6b"   # w1<0, w2<0
C_SE      = "#fb923c"   # w1>0, w2<0
C_CORNER  = "#ffdd57"   # sparse corners

T   = 1.5   # λ
LIM = 2.8

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 9), facecolor=BG)
ax.set_facecolor(BG)
for sp in ax.spines.values():
    sp.set_edgecolor("#3a3a3a")
ax.tick_params(colors="white", labelsize=10)

# ── Diamond fill ──────────────────────────────────────────────────────────────
verts = np.array([[T, 0], [0, T], [-T, 0], [0, -T]])
ax.add_patch(plt.Polygon(verts, closed=True,
                         facecolor=BLUE, alpha=0.25, edgecolor="none", zorder=2))

# ── 4 coloured edges ──────────────────────────────────────────────────────────
lw = 5
# NE: w1>0, w2>0
ax.plot([T, 0], [0, T],   color=C_NE, lw=lw, zorder=5, solid_capstyle="round")
# NW: w1<0, w2>0
ax.plot([0, -T], [T, 0],  color=C_NW, lw=lw, zorder=5, solid_capstyle="round")
# SW: w1<0, w2<0
ax.plot([-T, 0], [0, -T], color=C_SW, lw=lw, zorder=5, solid_capstyle="round")
# SE: w1>0, w2<0
ax.plot([0, T], [-T, 0],  color=C_SE, lw=lw, zorder=5, solid_capstyle="round")

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.axhline(0, color="#3a3a3a", lw=0.8, zorder=1)
ax.axvline(0, color="#3a3a3a", lw=0.8, zorder=1)

# ── Interior label ────────────────────────────────────────────────────────────
ax.text(0, 0, "$w = 0$\n$r_1,r_2\\in[-\\lambda,\\lambda]$",
        color="#c8c8ff", fontsize=8.5, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e1e3a", edgecolor="#7777aa", alpha=0.92))

# ── Sparse corners — boxes placed outside diamond along outward axis ───────────
#  (cx, cy) = corner dot,  (tx, ty) = text anchor,  ha/va = text alignment
corners = [
    ( T,  0, "$w_2=0$\n$r_1{=}+\\lambda$\n$r_2\\in[-\\lambda,\\lambda]$",  T+0.10,       0,   "left",   "center"),
    (-T,  0, "$w_2=0$\n$r_1{=}-\\lambda$\n$r_2\\in[-\\lambda,\\lambda]$", -T-0.10,       0,   "right",  "center"),
    ( 0,  T, "$w_1=0$\n$r_2{=}+\\lambda$\n$r_1\\in[-\\lambda,\\lambda]$",       0,  T+0.10,  "center",  "bottom"),
    ( 0, -T, "$w_1=0$\n$r_2{=}-\\lambda$\n$r_1\\in[-\\lambda,\\lambda]$",       0, -T-0.10,  "center",  "top"),
]
for (cx, cy, lbl, tx, ty, ha, va) in corners:
    ax.scatter(cx, cy, color=C_CORNER, s=130, zorder=8,
               edgecolors="white", linewidths=1.2)
    ax.annotate(lbl,
                xy=(cx, cy), xytext=(tx, ty),
                xycoords="data", textcoords="data",
                color=C_CORNER, fontsize=8.2, ha=ha, va=va,
                arrowprops=dict(arrowstyle="-", color=C_CORNER, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                          edgecolor=C_CORNER, alpha=0.9))

# ── Styling ───────────────────────────────────────────────────────────────────
ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_aspect("equal")
ax.set_xlabel("$w_1$", color="white", fontsize=14)
ax.set_ylabel("$w_2$", color="white", fontsize=14, rotation=0, labelpad=16)
ax.set_title("Conditions on the L1 Circle\n"
             r"$r_i \equiv x_i^T(y - Xw^*)$ — the residual correlation for feature $i$",
             color="white", fontsize=12, fontweight="bold", pad=12)

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=C_NE,     label=r"$w_1>0,\;w_2>0$ — Case 2 for both"),
    mpatches.Patch(color=C_NW,     label=r"$w_1<0,\;w_2>0$ — Case 1 & 2"),
    mpatches.Patch(color=C_SW,     label=r"$w_1<0,\;w_2<0$ — Case 1 for both"),
    mpatches.Patch(color=C_SE,     label=r"$w_1>0,\;w_2<0$ — Case 2 & 1"),
    mpatches.Patch(color=C_CORNER, label=r"Corner: one weight $=0$ → Case 3 (sparse)"),
]
ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.21),
          ncol=2, facecolor="#161b27", edgecolor="#2a2a4a",
          labelcolor="white", fontsize=8.8,
          title="Edge colour = which KKT case holds at that point on the boundary",
          title_fontsize=8.5)

plt.tight_layout()
out = OUTPUT_DIR / f"{stamp}_lasso_kkt_conditions.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
plt.close(fig)
