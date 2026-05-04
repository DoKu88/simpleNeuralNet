"""
lasso_ridge_high_dim_paths.py

Regularization paths for n=10 weights — L1 Lasso vs L2 Ridge.
Shows induced sparsity: Lasso zeros weights one-by-one while Ridge
shrinks all uniformly.  Bottom panels quantify the efficiency story.

Assumes orthogonal features (X^TX = I) for clean closed-form paths:
  Lasso : w_i(λ) = max(0,  |w_OLS_i| − λ/2)   (soft-thresholding)
  Ridge : w_i(λ) = w_OLS_i / (1 + λ)           (proportional shrinkage)
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# ══════════════════════════════════════════════════════════════════════════════
#  OLS WEIGHTS  — a few large "signal" weights, many small "noise" weights
# ══════════════════════════════════════════════════════════════════════════════
W_OLS = np.array([3.0, 2.2, 1.8, 1.2, 0.9, 0.6, 0.4, 0.25, 0.15, 0.08])
n     = len(W_OLS)

LAM_MAX = 7.0
lam     = np.linspace(0, LAM_MAX, 1400)

# ── Regularization paths ──────────────────────────────────────────────────────
lasso_paths = np.maximum(0.0, W_OLS[:, None] - lam[None, :] / 2)   # (n, L)
ridge_paths = W_OLS[:, None] / (1.0 + lam[None, :])                 # (n, L)

# ── Lasso sparsity thresholds: λ_i = 2 |w_OLS_i| ────────────────────────────
zero_lams = 2.0 * W_OLS   # [6.0, 4.4, 3.6, 2.4, 1.8, 1.2, 0.8, 0.5, 0.3, 0.16]

# ── Active parameters vs λ ────────────────────────────────────────────────────
n_active = np.sum(lasso_paths > 1e-9, axis=0)   # (L,)

# ── Cumulative signal retained when top-k Lasso weights are kept ─────────────
norm_ols   = np.linalg.norm(W_OLS)
cum_signal = np.array([
    np.linalg.norm(W_OLS[:k]) / norm_ols * 100
    for k in range(1, n + 1)
])   # cum_signal[k-1] = % retained with k active weights

# ── Signal retained vs λ for both methods ────────────────────────────────────
signal_lasso = np.linalg.norm(lasso_paths, axis=0) / norm_ols * 100
signal_ridge = np.linalg.norm(ridge_paths, axis=0) / norm_ols * 100

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════════════════
# tab10 gives 10 perceptually distinct colours — one per weight
COLORS    = [plt.cm.tab10(i) for i in range(n)]
GRID_CLR  = "#ebebeb"
LW        = 2.2
LASSO_CLR = "#e74c3c"
RIDGE_CLR = "#2563eb"

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(
    2, 2, figure=fig,
    height_ratios=[3, 2.2],
    hspace=0.44, wspace=0.26,
    left=0.07, right=0.97, top=0.93, bottom=0.08,
)
ax_l  = fig.add_subplot(gs[0, 0])
ax_r  = fig.add_subplot(gs[0, 1])
ax_b  = fig.add_subplot(gs[1, :])
ax_b2 = ax_b.twinx()

fig.suptitle(
    rf"Lasso vs Ridge — Regularization Paths  ($n = {n}$ weights)",
    fontsize=15, fontweight="bold", y=0.99,
)


def _style(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.grid(True, color=GRID_CLR, zorder=0)
    ax.tick_params(labelsize=10)


# ── Shared weight legend (same for both path panels) ─────────────────────────
legend_handles = [
    Line2D([0], [0], color=COLORS[i], lw=2.2,
           label=rf"$w_{{{i+1}}}^{{\rm OLS}}={W_OLS[i]}$")
    for i in range(n)
]

# ── [0,0] Lasso weight paths ──────────────────────────────────────────────────
_style(ax_l)
for i in range(n - 1, -1, -1):
    ax_l.plot(lam, lasso_paths[i], color=COLORS[i], lw=LW, zorder=3 + i * 0.1)

ax_l.set_xlim(0, LAM_MAX)
ax_l.set_ylim(-0.1, W_OLS[0] * 1.08)
ax_l.set_xlabel(r"$\lambda$", fontsize=13)
ax_l.set_ylabel("Weight Value", fontsize=12)
ax_l.set_title("L1  Lasso — Weights Zero Out One By One",
               fontsize=12, fontweight="bold", pad=8)
ax_l.legend(handles=legend_handles, fontsize=7.5, ncol=2,
            loc="upper right", framealpha=0.9)

# ── [0,1] Ridge weight paths ──────────────────────────────────────────────────
_style(ax_r)
for i in range(n - 1, -1, -1):
    ax_r.plot(lam, ridge_paths[i], color=COLORS[i], lw=LW, zorder=3 + i * 0.1)

ax_r.set_xlim(0, LAM_MAX)
ax_r.set_ylim(-0.1, W_OLS[0] * 1.08)
ax_r.set_xlabel(r"$\lambda$", fontsize=13)
ax_r.set_title("L2  Ridge — All Weights Stay Nonzero",
               fontsize=12, fontweight="bold", pad=8)
ax_r.legend(handles=legend_handles, fontsize=7.5, ncol=2,
            loc="upper right", framealpha=0.9)

# ── [1, :] Active parameters (left axis) + Signal retained (right axis) ───────
_style(ax_b)

# Left axis — active parameters
ax_b.step(lam, n_active, where="post", color=LASSO_CLR,
          lw=2.5, label="Lasso — Active Parameters", zorder=4)
ax_b.axhline(n, color=RIDGE_CLR, lw=2.5, ls="-",
             label=f"Ridge — Active Parameters (always {n})", zorder=3, alpha=0.9)

ax_b.set_xlim(0, LAM_MAX)
ax_b.set_ylim(-0.3, n + 0.9)
ax_b.set_yticks(range(0, n + 1, 2))
ax_b.set_xlabel(r"$\lambda$", fontsize=13)
ax_b.set_ylabel("Active Parameters", fontsize=12)


# Right axis — signal retained (%)
ax_b2.plot(lam, signal_lasso, color=LASSO_CLR, lw=2.0, ls="--",
           label="Lasso — Signal Retained", zorder=4, alpha=0.85)
ax_b2.plot(lam, signal_ridge, color=RIDGE_CLR, lw=2.0, ls="--",
           label="Ridge — Signal Retained", zorder=3, alpha=0.85)

ax_b2.set_ylim(0, 108)
ax_b2.set_ylabel("Signal Retained (%)", fontsize=12)
ax_b2.tick_params(axis="y", labelsize=10)
for sp in ax_b2.spines.values():
    sp.set_color("#cccccc")

# Combined legend
h1, l1 = ax_b.get_legend_handles_labels()
h2, l2 = ax_b2.get_legend_handles_labels()
ax_b.legend(h1 + h2, l1 + l2, fontsize=10, framealpha=0.9, loc="upper right")

ax_b.set_title(r"# Active Parameters & Signal Retained (%) vs $\lambda$",
               fontsize=12, fontweight="bold", pad=8)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = OUTPUT_DIR / f"{stamp}_lasso_ridge_high_dim_paths.png"
fig.savefig(str(out_path), dpi=140, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.show()
