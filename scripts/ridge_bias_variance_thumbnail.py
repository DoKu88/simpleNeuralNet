"""
ridge_bias_variance_thumbnail.py

16:9 static PNG thumbnail for the bias-variance tradeoff blog post.
Single panel: Bias²(λ), Variance(λ), and Total error vs λ on a log scale,
with a highlighted optimal-λ marker and direct curve labels.

Same RBF model as ridge_bias_variance.py — only the figure differs.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_COL  = "#21262d"
TEXT_COL  = "#e6edf3"
MUTED_COL = "#8b949e"
BIAS_COL  = "#e74c3c"
VAR_COL   = "#3498db"
TOT_COL   = "#2ecc71"

# ── Model setup (identical to ridge_bias_variance.py) ────────────────────────────
def f_true(x):
    return np.sin(2.5 * x) + 0.4 * np.cos(5 * x)


NOISE_STD  = 0.5
N_TRAIN    = 20
N_DATASETS = 200
N_CENTERS  = 50
BANDWIDTH  = 0.25

rng = np.random.default_rng(42)

x_train = np.linspace(0, np.pi, N_TRAIN)
x_plot  = np.linspace(0, np.pi, 300)
centers = np.linspace(0, np.pi, N_CENTERS)


def rbf_matrix(x, c, bw):
    return np.exp(-0.5 * ((x[:, None] - c[None, :]) / bw) ** 2)


X_train = rbf_matrix(x_train, centers, BANDWIDTH)
X_plot  = rbf_matrix(x_plot,  centers, BANDWIDTH)

y_clean    = f_true(x_train)
Y_datasets = y_clean + rng.normal(0, NOISE_STD, (N_DATASETS, N_TRAIN))

XtX = X_train.T @ X_train
I_p = np.eye(N_CENTERS)


def ridge_preds_all(lam):
    Xty = X_train.T @ Y_datasets.T
    W   = np.linalg.solve(XtX + lam * I_p, Xty)
    return (X_plot @ W).T


# ── Compute bias² and variance over a dense λ grid ───────────────────────────────
y_true_plot   = f_true(x_plot)
N_DENSE       = 400
lambdas_dense = np.logspace(-4, 2, N_DENSE)
bias2_dense   = np.zeros(N_DENSE)
var_dense     = np.zeros(N_DENSE)

print("Computing bias/variance curves...")
for i, lam in enumerate(lambdas_dense):
    pd = ridge_preds_all(lam)
    bias2_dense[i] = ((pd.mean(axis=0) - y_true_plot) ** 2).mean()
    var_dense[i]   = pd.var(axis=0).mean()
total_dense = bias2_dense + var_dense
print("Done.")

opt_idx    = np.argmin(total_dense)
opt_lam    = lambdas_dense[opt_idx]
opt_total  = total_dense[opt_idx]

# ── Figure ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)
for sp in ax.spines.values():
    sp.set_color("#30363d")
ax.tick_params(colors=TEXT_COL, labelsize=11)

# Shaded fills
ax.fill_between(lambdas_dense, bias2_dense, alpha=0.10, color=BIAS_COL, zorder=1)
ax.fill_between(lambdas_dense, var_dense,   alpha=0.10, color=VAR_COL,  zorder=1)

# Main curves
ax.semilogx(lambdas_dense, bias2_dense, color=BIAS_COL, lw=3.5, zorder=3, label=r"Bias$^2(\lambda)$")
ax.semilogx(lambdas_dense, var_dense,   color=VAR_COL,  lw=3.5, zorder=3, label=r"Variance$(\lambda)$")
ax.semilogx(lambdas_dense, total_dense, color=TOT_COL,  lw=3.5, ls="--", zorder=3, label="Total Error")

# Optimal λ: vertical line + star
ax.axvline(opt_lam, color="white", lw=1.5, ls=":", alpha=0.55, zorder=2)
ax.plot(opt_lam, opt_total, "*", color="#f1c40f", ms=18,
        markeredgecolor="white", markeredgewidth=1.0, zorder=5,
        label=r"Optimal $\lambda$")


# Legend
ax.legend(fontsize=13, facecolor=PANEL_BG, edgecolor="#30363d",
          labelcolor=TEXT_COL, loc="upper left", framealpha=0.85)

# Axes
ax.set_xlabel(r"$\lambda$   (regularisation strength)", color=TEXT_COL, fontsize=16)
ax.set_ylabel("Expected Error", color=TEXT_COL, fontsize=16)
ax.set_xlim(lambdas_dense[0], lambdas_dense[-1])
ax.set_ylim(bottom=0)
ax.grid(True, color=GRID_COL, alpha=0.5, which="both")

# Title + subtitle
fig.text(0.5, 0.97,
         "Bias–Variance Tradeoff",
         ha="center", va="top", fontsize=30, fontweight="bold", color=TEXT_COL)
fig.text(0.5, 0.895,
         r"Ridge regression: $\hat{w}_\lambda = (X^\top X + \lambda I)^{-1} X^\top y$",
         ha="center", va="top", fontsize=18, color=TEXT_COL)

fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.12)

stamp    = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
out_path = OUTPUT_DIR / f"{stamp}_ridge_bias_variance_thumbnail.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print(f"Saved to: {out_path}")
plt.show()
