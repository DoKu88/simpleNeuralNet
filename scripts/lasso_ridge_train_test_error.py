"""
lasso_ridge_train_test_error.py

Train vs. test MSE across regularization strength for L1 Lasso and L2 Ridge.

Same weight magnitudes as lasso_ridge_high_dim_paths.py, but the TRUE model
is sparse — only the first 5 of 10 weights are real signal; the last 5 are
zero.  Lasso can zero out the noise weights and approach the irreducible noise
floor; Ridge must keep all weights and pays a bias penalty.

Requires: scikit-learn  (pip install scikit-learn)
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, Ridge

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# ══════════════════════════════════════════════════════════════════════════════
#  TRUE MODEL — sparse: last 5 weights are zero (noise in the OLS estimates)
# ══════════════════════════════════════════════════════════════════════════════
W_TRUE     = np.array([3.0, 2.2, 1.8, 1.2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
n_features = len(W_TRUE)
n_signal   = int(np.sum(W_TRUE != 0))   # 5

# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
n_train, n_test = 60, 3000
sigma = 0.8   # noise std

X_train = np.random.randn(n_train, n_features)
y_train = X_train @ W_TRUE + sigma * np.random.randn(n_train)
X_test  = np.random.randn(n_test,  n_features)
y_test  = X_test  @ W_TRUE + sigma * np.random.randn(n_test)

# ══════════════════════════════════════════════════════════════════════════════
#  SWEEP ALPHAS
# ══════════════════════════════════════════════════════════════════════════════
alphas = np.logspace(-2, 2, 300)

lasso_train_mse = np.empty(len(alphas))
lasso_test_mse  = np.empty(len(alphas))
ridge_train_mse = np.empty(len(alphas))
ridge_test_mse  = np.empty(len(alphas))

for k, a in enumerate(alphas):
    lm = Lasso(alpha=a, max_iter=20_000, fit_intercept=False)
    lm.fit(X_train, y_train)
    lasso_train_mse[k] = np.mean((y_train - lm.predict(X_train)) ** 2)
    lasso_test_mse[k]  = np.mean((y_test  - lm.predict(X_test))  ** 2)

    rm = Ridge(alpha=a, fit_intercept=False)
    rm.fit(X_train, y_train)
    ridge_train_mse[k] = np.mean((y_train - rm.predict(X_train)) ** 2)
    ridge_test_mse[k]  = np.mean((y_test  - rm.predict(X_test))  ** 2)

li = int(np.argmin(lasso_test_mse))
ri = int(np.argmin(ridge_test_mse))

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════════
LASSO_CLR = "#e74c3c"
RIDGE_CLR = "#2563eb"
GRID_CLR  = "#ebebeb"

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
for sp in ax.spines.values():
    sp.set_color("#cccccc")
ax.grid(True, color=GRID_CLR, zorder=0)
ax.tick_params(labelsize=10)

# ── Lines ─────────────────────────────────────────────────────────────────────
ax.semilogx(alphas, lasso_train_mse, color=LASSO_CLR, lw=1.8, ls="--",
            alpha=0.70, label="Lasso — Train", zorder=3)
ax.semilogx(alphas, lasso_test_mse,  color=LASSO_CLR, lw=2.5, ls="-",
            label="Lasso — Test",  zorder=4)
ax.semilogx(alphas, ridge_train_mse, color=RIDGE_CLR, lw=1.8, ls="--",
            alpha=0.70, label="Ridge — Train", zorder=3)
ax.semilogx(alphas, ridge_test_mse,  color=RIDGE_CLR, lw=2.5, ls="-",
            label="Ridge — Test",  zorder=4)

# ── Noise floor ───────────────────────────────────────────────────────────────
ax.axhline(sigma ** 2, color="#999", lw=1.2, ls=":",
           zorder=2, label=rf"Noise Floor ($\sigma^2 = {sigma**2:.2f}$)")

# ── Optimal test-MSE markers ──────────────────────────────────────────────────
for idx, mse, clr, side in [
    (li, lasso_test_mse[li], LASSO_CLR, "right"),
    (ri, ridge_test_mse[ri], RIDGE_CLR, "left"),
]:
    ax.scatter([alphas[idx]], [mse], color=clr, s=90, zorder=10,
               edgecolors="white", linewidths=0.9)
    xoff = alphas[idx] * (3.5 if side == "right" else 0.28)
    ax.annotate(
        f"α* = {alphas[idx]:.3f}\nTest MSE = {mse:.3f}",
        xy=(alphas[idx], mse),
        xytext=(xoff, mse + 0.55),
        fontsize=8.5, color=clr, ha=side,
        arrowprops=dict(arrowstyle="-|>", color=clr, lw=0.8, mutation_scale=8),
    )

# ── Labels ────────────────────────────────────────────────────────────────────
ax.set_xlabel(r"Regularization Strength ($\alpha$)", fontsize=13)
ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12)
ax.set_title(
    rf"Train vs. Test Error — L1 Lasso vs L2 Ridge"
    rf"  ($n_\mathrm{{train}}={n_train}$, $p={n_features}$ features,"
    rf" {n_signal} true nonzero weights)",
    fontsize=12, fontweight="bold",
)
ax.legend(fontsize=10, framealpha=0.9)

plt.tight_layout()
out_path = OUTPUT_DIR / f"{stamp}_lasso_ridge_train_test_error.png"
fig.savefig(str(out_path), dpi=140, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.show()
