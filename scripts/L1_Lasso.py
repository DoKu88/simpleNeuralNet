"""
L1_Lasso.py

Animated GIF for Lasso (L1) regularisation.

  2-D: classic diamond constraint vs. elliptical loss contours.
       λ cycles large → 0 → large (seamless loop), showing the constrained
       minimum snap to a corner as the L1 ball shrinks.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# ── 2-D loss: tilted elliptical bowl (minimum outside the origin) ─────────────
W_OPT_2 = np.array([1.9, 1.6])
A2, B2  = 0.55, 1.15

def loss2d(w0, w1):
    return (w0 - W_OPT_2[0])**2 / A2 + (w1 - W_OPT_2[1])**2 / B2

# ── Diamond path (|w0|+|w1| = t) ─────────────────────────────────────────────
def diamond_pts(t, n=500):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r   = 1.0 / (np.abs(np.cos(ang)) + np.abs(np.sin(ang)) + 1e-14)
    return np.column_stack([t * r * np.cos(ang), t * r * np.sin(ang)])

# ── 2-D constrained minimum ───────────────────────────────────────────────────
def constrained_min_2d(t):
    """argmin_{|w0|+|w1|≤t} loss2d"""
    if abs(W_OPT_2[0]) + abs(W_OPT_2[1]) <= t:
        return W_OPT_2.copy()
    best_L, best_w = np.inf, np.array([t, 0.0])
    n = 5000
    for s0, s1 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        lam = np.linspace(0, t, n)
        Ls  = loss2d(s0 * lam, s1 * (t - lam))
        idx = int(Ls.argmin())
        if Ls[idx] < best_L:
            best_L = Ls[idx]
            best_w = np.array([s0 * lam[idx], s1 * (t - lam[idx])])
    return best_w


# =============================================================================
# GIF 1 – 2-D
# =============================================================================
LIM2 = 2.8
w0v  = np.linspace(-LIM2, LIM2, 300)
w1v  = np.linspace(-LIM2, LIM2, 300)
W0G, W1G = np.meshgrid(w0v, w1v)
LG2      = loss2d(W0G, W1G)

# Cosine wave: large → zero → large, seamless loop (last frame → first frame).
_N2 = 160
_phase = np.linspace(0, 2 * np.pi, _N2, endpoint=False)
T_SEQ  = 3.6 * (1 + np.cos(_phase)) / 2

fig2, ax2 = plt.subplots(figsize=(7, 7), facecolor="white")
ax2.set_facecolor("white")

lev2 = np.linspace(0.0, 16.0, 26)
ax2.contourf(W0G, W1G, LG2, levels=lev2, cmap="YlOrRd_r", alpha=0.70)
ax2.contour( W0G, W1G, LG2, levels=lev2[::2], colors="black",
             linewidths=0.4, alpha=0.22)

ax2.scatter([W_OPT_2[0]], [W_OPT_2[1]], color="black", s=160,
            marker="*", zorder=10, label=r"Unconstrained $w_{\mathrm{optimal}}$")
ax2.axhline(0, color="black", lw=0.4, alpha=0.25)
ax2.axvline(0, color="black", lw=0.4, alpha=0.25)

ax2.set_xlabel("$w_0$", fontsize=13, color="black")
ax2.set_ylabel("$w_1$", fontsize=13, color="black")
ax2.tick_params(colors="black", labelsize=9)
for sp in ax2.spines.values():
    sp.set_edgecolor("#aaa")
ax2.set_xlim(-LIM2, LIM2)
ax2.set_ylim(-LIM2, LIM2)
ax2.set_aspect("equal")
ax2.set_title(
    "Lasso (L1)\n"
    r"$\mathcal{L}(w) = \|y - Xw\|^2 + \lambda\sum_i|w_i|$",
    fontsize=13, fontweight="bold", color="black", pad=10,
)

dpoly = MplPolygon(
    diamond_pts(T_SEQ[0]), closed=True,
    facecolor="#2563eb", edgecolor="#1d4ed8",
    linewidth=2.0, alpha=0.22, zorder=4, label="L1 ball",
)
ax2.add_patch(dpoly)

cpt2, = ax2.plot([], [], "o", color="#dc2626", ms=11,
                 markeredgecolor="white", mew=0.8,
                 zorder=8, label="Constrained min")
tlbl2 = ax2.text(0.03, 0.04, rf"$\lambda = {T_SEQ[0]:.2f}$",
                 transform=ax2.transAxes, fontsize=12,
                 color="black", va="bottom")

ax2.legend(fontsize=9, loc="upper right", facecolor="white",
           edgecolor="#aaa", labelcolor="black", framealpha=0.90)


def anim2d(i):
    t = T_SEQ[i]
    dpoly.set_xy(diamond_pts(t))
    cw = constrained_min_2d(t)
    cpt2.set_data([cw[0]], [cw[1]])
    tlbl2.set_text(rf"$\lambda = {t:.2f}$")


ani2 = animation.FuncAnimation(fig2, anim2d, frames=len(T_SEQ), interval=50)
gif2 = OUTPUT_DIR / f"{stamp}_lasso_2d.gif"
ani2.save(gif2, writer="pillow", fps=20, dpi=130)
print(f"Saved: {gif2}")
plt.close(fig2)
