import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.linear_model import LinearRegression

# ── style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.size": 13,
})

C0, C1 = "#0072B2", "#D55E00"

# ── data ──────────────────────────────────────────────────────────────────────

np.random.seed(42)
X, y = make_circles(n_samples=300, noise=0.08, factor=0.42)
y_signed = 2 * y - 1  # {-1, +1}

# ── basis function: phi(x) = r^2 = x1^2 + x2^2 ───────────────────────────────

r_sq = X[:, 0] ** 2 + X[:, 1] ** 2

# fit 1D linear regression on r^2 alone — trivially separable
reg = LinearRegression()
reg.fit(r_sq.reshape(-1, 1), y_signed)
accuracy = np.mean(np.sign(reg.predict(r_sq.reshape(-1, 1))) == y_signed)

# decision threshold in r^2 space: w * r^2 + b = 0
threshold = -reg.intercept_ / reg.coef_[0]

# ── figure: two panels side by side ───────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
fig.suptitle(
    r"Non-linear Data $\overset{\phi(\mathbf{x})}{\longrightarrow}$ Linearly Separable",
    fontsize=16, fontweight="bold",
)

# ── Panel 1: original 2D data ─────────────────────────────────────────────────

ax1.scatter(X[y == 0, 0], X[y == 0, 1], c=C0, label="Class 0 (inner)",
            alpha=0.75, edgecolors="k", linewidths=0.4, s=35)
ax1.scatter(X[y == 1, 0], X[y == 1, 1], c=C1, label="Class 1 (outer)",
            alpha=0.75, edgecolors="k", linewidths=0.4, s=35)
ax1.set_title("Original Data  (not linearly separable)", fontsize=13)
ax1.set_xlabel(r"$x_1$")
ax1.set_ylabel(r"$x_2$")
ax1.set_aspect("equal")
ax1.legend(fontsize=11, loc="lower left", framealpha=0.85)
ax1.grid(True, alpha=0.3)

# ── Panel 2: transformed space — r^2 on x-axis, jitter on y-axis ─────────────

rng = np.random.default_rng(7)
jitter = rng.uniform(-0.4, 0.4, size=len(y))

ax2.scatter(r_sq[y == 0], jitter[y == 0], c=C0, label="Class 0 (inner)",
            alpha=0.75, edgecolors="k", linewidths=0.4, s=35)
ax2.scatter(r_sq[y == 1], jitter[y == 1], c=C1, label="Class 1 (outer)",
            alpha=0.75, edgecolors="k", linewidths=0.4, s=35)

ax2.axvline(threshold, color="k", linewidth=2.5, linestyle="--", label=f"Decision boundary")

ax2.set_title(
    r"After Basis Function $\phi(\mathbf{x}) = x_1^2 + x_2^2$  (linearly separable)",
    fontsize=13,
)
ax2.set_xlabel(r"$\phi(\mathbf{x}) = x_1^2 + x_2^2$")
ax2.set_yticks([])
ax2.legend(fontsize=11, loc="lower left", framealpha=0.85)
ax2.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("outputs/non_linear_data_mapping.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Decision threshold (r^2): {threshold:.4f}")
print(f"Classification accuracy:  {accuracy:.1%}")
