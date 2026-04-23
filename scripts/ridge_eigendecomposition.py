"""
ridge_eigendecomposition.py

Ridge Regression through the lens of eigendecomposition of X^T X.

    L(w) = ||y - Xw||^2 + λ||w||^2
    w*(λ) = (X^T X + λI)^{-1} X^T y
          = Q (Λ + λI)^{-1} Q^T X^T y          [via X^T X = Q Λ Q^T]

Diagonal entries of (Λ + λI)^{-1}:   1 / (λ_i + λ)
Per-eigendirection shrinkage factor:  λ_i / (λ_i + λ)

Four panels:
  [0,0]  Effective inverse diagonal  1/(λ_i + λ)  vs λ  (log-log)
  [0,1]  Shrinkage factors  λ_i/(λ_i + λ)  vs λ  (semi-log x)
  [1,0]  Solution path  w*(λ)  in weight space  (with OLS loss contours)
  [1,1]  Fitted predictions on data for select λ  (projected onto PC₁)
"""

from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Synthetic data ─────────────────────────────────────────────────────────────
# Highly correlated features give X^T X a large eigenvalue ratio, making the
# differential shrinkage between eigendirections clearly visible.
rng = np.random.default_rng(7)
n   = 80

x1 = rng.normal(0, 1, n)
x2 = 0.90 * x1 + rng.normal(0, 0.44, n)
X  = np.column_stack([x1, x2])

w_true = np.array([1.5, -0.8])
y      = X @ w_true + rng.normal(0, 0.6, n)

# Center (removes intercept)
X -= X.mean(axis=0)
y -= y.mean()

# ── Eigendecomposition  X^T X = Q Λ Q^T ───────────────────────────────────────
XtX = X.T @ X                             # (2, 2)
Xty = X.T @ y

eigenvalues, Q = np.linalg.eigh(XtX)      # ascending: [λ_small, λ_large]

# OLS solution and its components in the eigenbasis
w_ols     = np.linalg.solve(XtX, Xty)
alpha_ols = Q.T @ w_ols                    # α = Q^T w_ols

# ── Ridge solutions ────────────────────────────────────────────────────────────
lambdas   = np.logspace(-2, 2.7, 500)
SHOW_LAMS = [0.0, 0.5, 2.0, 10.0, 50.0]
COLORS    = ["#1a1a2e", "#27ae60", "#2980b9", "#e67e22", "#e74c3c"]


def ridge(lam):
    return np.linalg.solve(XtX + lam * np.eye(2), Xty)


w_path = np.array([ridge(l) for l in lambdas])   # (500, 2)
w_show = {l: (ridge(l) if l > 0 else w_ols.copy()) for l in SHOW_LAMS}

# Effective inverse diagonals and shrinkage factors for each λ
inv_diag = np.stack([1.0 / (ev + lambdas) for ev in eigenvalues], axis=1)   # (500, 2)
shrink   = np.stack([ev / (ev + lambdas)  for ev in eigenvalues], axis=1)   # (500, 2)

# ── OLS loss landscape for the solution-path panel ────────────────────────────
pad    = 2.5
w1_lin = np.linspace(w_ols[0] - pad, w_ols[0] + pad, 200)
w2_lin = np.linspace(w_ols[1] - pad, w_ols[1] + pad, 200)
W1g, W2g = np.meshgrid(w1_lin, w2_lin)
W_flat   = np.stack([W1g.ravel(), W2g.ravel()], axis=1)      # (40000, 2)
resid_all = y[:, None] - X @ W_flat.T                         # (n, 40000)
LOSS_grid = np.sum(resid_all ** 2, axis=0).reshape(W1g.shape)
loss_ols  = float(np.sum((y - X @ w_ols) ** 2))

# ── Eigenvector colors / labels ────────────────────────────────────────────────
EI_COLORS = ["#c0392b", "#2980b9"]
EI_LABELS = [
    rf"$\lambda_{{small}} = {eigenvalues[0]:.1f}$  (weak direction)",
    rf"$\lambda_{{large}} = {eigenvalues[1]:.1f}$  (dominant direction)",
]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    r"Ridge Regression via Eigendecomposition  —  "
    r"$w^*(\lambda) = Q\,(\Lambda + \lambda I)^{-1}Q^\top X^\top y$",
    fontsize=14, fontweight="bold",
)

# ── [0,0]: Effective inverse eigenvalues  1/(λ_i + λ) ─────────────────────────
ax = axes[0, 0]

for j in range(2):
    ax.loglog(lambdas, inv_diag[:, j], color=EI_COLORS[j], linewidth=2.5,
              label=EI_LABELS[j])
    # Reference line: constant 1/λ_i (OLS limit)
    ax.axhline(1.0 / eigenvalues[j], color=EI_COLORS[j],
               linewidth=0.9, linestyle=":", alpha=0.45)

# Where the two curves merge into the same 1/λ line
lam_merge = eigenvalues[1] * 3
ax.axvline(lam_merge, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
ax.text(lam_merge * 1.1, inv_diag[:, 1].max() * 0.6,
        r"curves merge $\approx 1/\lambda$", fontsize=8, color="grey", va="top")

# Mark the specific λ values used in other panels
for lam, col in zip(SHOW_LAMS[1:], COLORS[1:]):
    ax.axvline(lam, color=col, linewidth=0.7, linestyle=":", alpha=0.5, zorder=1)

ax.set_xlabel(r"Regularisation  $\lambda$  (log scale)", fontsize=11)
ax.set_ylabel(r"$1 / (\lambda_i + \lambda)$  (log scale)", fontsize=11)
ax.set_title(
    r"Effective Diagonal of $(\Lambda + \lambda I)^{-1}$"
    "\n"
    r"Small eigenvalue is suppressed first — weak directions penalised most",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

# ── [0,1]: Per-eigendirection shrinkage  λ_i/(λ_i + λ) ───────────────────────
ax = axes[0, 1]

for j in range(2):
    ax.semilogx(lambdas, shrink[:, j], color=EI_COLORS[j], linewidth=2.5,
                label=EI_LABELS[j])

# Annotate differential gap at a middle λ
lam_mid = 4.0
sf = eigenvalues / (eigenvalues + lam_mid)
ax.annotate("",
            xy=(lam_mid, sf[0]), xytext=(lam_mid, sf[1]),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.5))
ax.text(lam_mid * 1.4, (sf[0] + sf[1]) / 2 + 0.02,
        "differential\nshrinkage", fontsize=8, color="#555555", va="center")

# Mark specific λ values with scatter dots
for lam, col in zip(SHOW_LAMS[1:], COLORS[1:]):
    sf_lam = eigenvalues / (eigenvalues + lam)
    ax.scatter([lam, lam], sf_lam, color=col, s=60, zorder=5)
    ax.axvline(lam, color=col, linewidth=0.7, linestyle=":", alpha=0.5, zorder=1)

ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5,
           label=r"OLS limit  ($\lambda = 0$)")
ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5,
           label=r"Zero limit  ($\lambda \to \infty$)")

ax.set_ylim(-0.06, 1.12)
ax.set_xlabel(r"Regularisation  $\lambda$  (log scale)", fontsize=11)
ax.set_ylabel(r"Shrinkage factor  $\lambda_i / (\lambda_i + \lambda)$", fontsize=11)
ax.set_title(
    r"Per-eigendirection Shrinkage Factor"
    "\n"
    r"1 = unchanged (OLS),  0 = fully shrunk to zero",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── [1,0]: Solution path  w*(λ)  in weight space ──────────────────────────────
ax = axes[1, 0]

# OLS loss contours as background reference
contour_levels = loss_ols + np.array([1, 4, 12, 30, 65, 120])
ax.contourf(W1g, W2g, LOSS_grid,
            levels=np.linspace(LOSS_grid.min(), LOSS_grid.max(), 35),
            cmap="Blues", alpha=0.28, zorder=0)
ax.contour(W1g, W2g, LOSS_grid, levels=contour_levels,
           colors="#2060a8", linewidths=0.9, alpha=0.5, zorder=1)

# Regularization path colored by log10(λ)
cmap_path = cm.get_cmap("plasma")
norm_path  = plt.Normalize(np.log10(lambdas[0]), np.log10(lambdas[-1]))
for i in range(len(lambdas) - 1):
    c = cmap_path(norm_path(np.log10(lambdas[i])))
    ax.plot(w_path[i:i+2, 0], w_path[i:i+2, 1], color=c, linewidth=2.5, zorder=3)

# Mark the OLS solution and specific λ solutions
for (lam, w), col in zip(w_show.items(), COLORS):
    mk = "*" if lam == 0.0 else "o"
    sz = 160 if lam == 0.0 else 100
    ax.scatter(*w, color=col, s=sz, marker=mk, zorder=8,
               edgecolor="white", linewidth=1.3)
    ax.text(w[0] + 0.10, w[1] + 0.10,
            r"OLS $(\lambda=0)$" if lam == 0.0 else rf"$\lambda={lam}$",
            fontsize=8.5, color=col, fontweight="bold", zorder=9)

# Origin: λ→∞ limit
ax.scatter(0, 0, color="grey", s=80, marker="x", linewidth=2.5, zorder=6)
ax.text(0.09, 0.09, r"$\lambda \to \infty$", fontsize=8.5, color="grey")

sm = cm.ScalarMappable(cmap=cmap_path, norm=norm_path)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.04)
cbar.set_label(r"$\log_{10}\lambda$", fontsize=10)

ax.set_xlabel(r"$w_1$", fontsize=12)
ax.set_ylabel(r"$w_2$", fontsize=12)
ax.set_title(
    r"Solution Path  $w^*(\lambda)$  in Weight Space"
    "\n"
    r"Path shrinks from OLS toward the origin as $\lambda$ grows",
    fontsize=11,
)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2, zorder=0)

# ── [1,1]: Fitted predictions projected onto PC₁ ──────────────────────────────
ax = axes[1, 1]

# PC₁ = eigenvector for the largest eigenvalue (index 1, ascending order)
pc1  = Q[:, 1]
z    = X @ pc1    # 1-D projection of each data point

ax.scatter(z, y, s=18, color="steelblue", alpha=0.45, zorder=2, label="Data")

z_plot   = np.linspace(z.min(), z.max(), 200)
X_proj   = np.outer(z_plot, pc1)   # synthetic X along PC₁

for (lam, w), col in zip(w_show.items(), COLORS):
    y_pred = X_proj @ w
    lbl    = (r"OLS $(\lambda=0)$" if lam == 0.0 else rf"$\lambda={lam}$") + \
             rf"  $\|w\|={np.linalg.norm(w):.2f}$"
    lw     = 3.0 if lam == 0.0 else 2.0
    ax.plot(z_plot, y_pred, color=col, linewidth=lw, label=lbl, zorder=4)

ax.set_xlabel(r"Projection onto PC$_1$  (dominant eigendirection)", fontsize=11)
ax.set_ylabel(r"$y$", fontsize=12)
ax.set_title(
    r"Fitted Predictions Along Dominant Eigendirection"
    "\n"
    r"Larger $\lambda$ shrinks $\|w\|$ and flattens the fit",
    fontsize=11,
)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(True, alpha=0.2)

# ── Save ───────────────────────────────────────────────────────────────────────
fig.tight_layout(rect=[0, 0, 1, 0.96])
stamp    = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
out_path = OUTPUT_DIR / f"{stamp}_ridge_eigendecomposition.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to: {out_path}")
plt.show()
