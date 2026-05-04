"""
lasso_ridge_dataset_overview.py

Shows the synthetic dataset underlying lasso_ridge_high_dim_paths.py.

Two panels:
  [left]   True weights w_OLS — bar chart (few large signals, many small)
  [right]  Fitted vs observed y — scatter, confirms regression signal
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# ══════════════════════════════════════════════════════════════════════════════
#  SAME WEIGHTS as lasso_ridge_high_dim_paths.py
# ══════════════════════════════════════════════════════════════════════════════
W_OLS      = np.array([3.0, 2.2, 1.8, 1.2, 0.9, 0.6, 0.4, 0.25, 0.15, 0.08])
n_features = len(W_OLS)

# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA
#  QR decomposition gives exactly orthonormal columns: Q^T Q = I
#  OLS formula w_hat = X^T y then recovers W_OLS + small noise
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
n_samples = 50
sigma     = 0.5

X_raw = np.random.randn(n_samples, n_features)
Q, _  = np.linalg.qr(X_raw)   # Q: (50×10), Q^T Q = I exactly
X     = Q

y     = X @ W_OLS + sigma * np.random.randn(n_samples)
y_hat = X @ W_OLS

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════════════════
COLORS   = [plt.cm.tab10(i) for i in range(n_features)]
GRID_CLR = "#ebebeb"

def _style(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.grid(True, color=GRID_CLR, zorder=0)
    ax.tick_params(labelsize=10)

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax_w, ax_y) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.13, wspace=0.30)

fig.suptitle(
    r"Dataset Overview — Synthetic Orthogonal Regression   "
    r"$X^\top\!X = I$,   "
    r"$\varepsilon \sim \mathcal{N}(0,\,\sigma^2)$,   "
    rf"$\sigma={sigma}$",
    fontsize=11, fontweight="bold", y=0.99,
)
fig.text(
    0.5, 0.93,
    rf"$y = Xw^{{\mathrm{{OLS}}}} + \varepsilon"
    rf"\qquad y \in \mathbb{{R}}^{{{n_samples}}},\;"
    rf"X \in \mathbb{{R}}^{{{n_samples} \times {n_features}}},\;"
    rf"w \in \mathbb{{R}}^{{{n_features}}},\;"
    rf"\varepsilon \in \mathbb{{R}}^{{{n_samples}}}$",
    ha="center", va="top", fontsize=13,
)

# ── True weights bar chart ────────────────────────────────────────────────────
_style(ax_w)
ax_w.bar(np.arange(1, n_features + 1), W_OLS,
         color=COLORS, alpha=0.85, zorder=3, edgecolor="white", linewidth=0.5)
for i, v in enumerate(W_OLS):
    ax_w.text(i + 1, v + 0.06, str(v),
              ha="center", va="bottom", fontsize=8, color="#444")
ax_w.set_xlabel("Feature Index", fontsize=12)
ax_w.set_ylabel("Weight Value", fontsize=12)
ax_w.set_title(r"True Weights $w^{\mathrm{OLS}}$  (starting point at $\lambda = 0$)",
               fontsize=11, fontweight="bold", pad=8)
ax_w.set_xticks(np.arange(1, n_features + 1))
ax_w.set_ylim(0, W_OLS[0] * 1.22)

# ── Fitted vs observed scatter ────────────────────────────────────────────────
_style(ax_y)
ax_y.scatter(y_hat, y, color="#6366f1", alpha=0.75, s=48, zorder=3,
             edgecolors="white", linewidths=0.5, label="Samples")
lo = min(y_hat.min(), y.min()) - 0.4
hi = max(y_hat.max(), y.max()) + 0.4
ax_y.plot([lo, hi], [lo, hi], color="#888", lw=1.4, ls="--", zorder=2, label="Perfect fit")
ax_y.set_xlabel(r"Fitted  $\hat{y} = Xw^{\mathrm{OLS}}$", fontsize=12)
ax_y.set_ylabel(r"Observed  $y$", fontsize=12)
ax_y.set_title(rf"Fitted vs Observed  ($\sigma = {sigma}$ noise)",
               fontsize=11, fontweight="bold", pad=8)
ax_y.legend(fontsize=10, framealpha=0.9)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = OUTPUT_DIR / f"{stamp}_lasso_ridge_dataset_overview.png"
fig.savefig(str(out_path), dpi=140, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.show()
