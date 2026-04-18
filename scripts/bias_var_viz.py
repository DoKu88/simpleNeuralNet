import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Ground truth: deliberately wiggly function.
# Kept continuous/mostly smooth so model families show a clear bias/variance
# trade-off, while oscillations make the individual fitted curves more obvious.
def ground_truth(x):
    # Base 5th-degree polynomial.
    base = x + 0.8 * x**2 - x**3 + 0.3 * x**4 - 10 

    # Make endpoints and local extrema more extreme by:
    # 1) adding edge-weighted polynomial terms (peaks at x≈0 and x≈6),
    # 2) increasing oscillation amplitude near the edges.
    x = np.asarray(x)
    centered = x - 3.0
    norm = centered / 3.0  # norm in [-1, 1] for x in [0, 6]

    edge_even = 5.0 * (norm**6)    # same sign at both ends
    edge_odd = 10.0 * (norm**5)    # flips sign between left/right

    edge_envelope = 1.0 + 1.2 * (norm**2)  # larger oscillations near edges

    osc1 = 1.4 * np.sin(2.2 * x + 0.6) * edge_envelope
    osc2 = 0.85 * np.sin(5.1 * x - 0.2) * edge_envelope

    return base + edge_even + edge_odd + osc1 + osc2

# Sample 100 points from the ground truth (no noise — we want pure bias/variance
# from model complexity, not data noise)
np.random.seed(42)
N_TOTAL = 15
N_TRAIN = 10
x_all = np.linspace(0, 6, N_TOTAL)
y_all = ground_truth(x_all)

# Dense x for plotting
x_plot = np.linspace(0, 6, 500)
y_true_plot = ground_truth(x_plot)

# All combinations of 10 indices from 100 — too many (100C10 ~ 1.7e13),
# so we randomly sample a large number of subsets instead
N_SUBSETS = 3000
rng = np.random.default_rng(0)
subsets = [rng.choice(N_TOTAL, size=N_TRAIN, replace=False) for _ in range(N_SUBSETS)]

def fit_and_predict(model_fn, subsets, x_all, y_all, x_plot):
    """Fit model on each subset, return predictions at x_plot: shape (N_SUBSETS, len(x_plot))."""
    preds = np.zeros((len(subsets), len(x_plot)))
    for i, idx in enumerate(subsets):
        X_train = x_all[idx].reshape(-1, 1)
        y_train = y_all[idx]
        model = model_fn()
        model.fit(X_train, y_train)
        preds[i] = model.predict(x_plot.reshape(-1, 1))
    return preds

# Model factories
def linear_model():
    return LinearRegression()

def poly5_model():
    return make_pipeline(PolynomialFeatures(degree=5), LinearRegression())

def poly10_model():
    return make_pipeline(PolynomialFeatures(degree=10), LinearRegression())

# Fit all subsets
preds_linear = fit_and_predict(linear_model, subsets, x_all, y_all, x_plot)
preds_poly5  = fit_and_predict(poly5_model,  subsets, x_all, y_all, x_plot)
preds_poly10 = fit_and_predict(poly10_model, subsets, x_all, y_all, x_plot)

# Bias and variance at each x
mean_linear = preds_linear.mean(axis=0)
mean_poly5  = preds_poly5.mean(axis=0)
mean_poly10 = preds_poly10.mean(axis=0)

bias2_linear = (mean_linear - y_true_plot) ** 2
bias2_poly5  = (mean_poly5  - y_true_plot) ** 2
bias2_poly10 = (mean_poly10 - y_true_plot) ** 2

var_linear = preds_linear.var(axis=0)
var_poly5  = preds_poly5.var(axis=0)
var_poly10 = preds_poly10.var(axis=0)

# Scalar summaries (integrated over x)
scalar_bias_linear = bias2_linear.mean()
scalar_var_linear  = var_linear.mean()
scalar_bias_poly5  = bias2_poly5.mean()
scalar_var_poly5   = var_poly5.mean()
scalar_bias_poly10 = bias2_poly10.mean()
scalar_var_poly10  = var_poly10.mean()

# Percentile bands for variance mass
def percentile_band(preds, lo=5, hi=95):
    return np.percentile(preds, lo, axis=0), np.percentile(preds, hi, axis=0)

lo_lin, hi_lin = percentile_band(preds_linear)
lo_p5,  hi_p5  = percentile_band(preds_poly5)
lo_p10, hi_p10 = percentile_band(preds_poly10)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.suptitle("Bias–Variance Trade-off", fontsize=15, fontweight="bold", y=0.98)

# 2 on top, 1 same-size centered on bottom
# GridSpec(2, 4): each top plot spans 2 cols; bottom spans cols 1-3 (centered)
gs = fig.add_gridspec(2, 4, hspace=0.25, wspace=0.3,
                      top=0.93, bottom=0.12, left=0.06, right=0.94)
ax_top_l = fig.add_subplot(gs[0, 0:2])
ax_top_r = fig.add_subplot(gs[0, 2:4])
ax_bot   = fig.add_subplot(gs[1, 1:3])

# Use robust y-limits so occasional unstable fits don't blow up the scale.
all_preds = np.vstack([preds_linear, preds_poly5, preds_poly10])
y_lo = min(y_true_plot.min(), np.percentile(all_preds, 1))
y_hi = max(y_true_plot.max(), np.percentile(all_preds, 99))
pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0
y_lim = (y_lo - pad, y_hi + pad)

for ax, preds, mean_pred, lo, hi, title, bias_val, var_val in [
    (ax_top_l, preds_linear, mean_linear, lo_lin, hi_lin,
     "Linear Regression (Degree 1)", scalar_bias_linear, scalar_var_linear),
    (ax_top_r, preds_poly5, mean_poly5, lo_p5, hi_p5,
     "Polynomial Regression (Degree 5)", scalar_bias_poly5, scalar_var_poly5),
    (ax_bot, preds_poly10, mean_poly10, lo_p10, hi_p10,
     "Polynomial Regression (Degree 10)", scalar_bias_poly10, scalar_var_poly10),
]:
    # Grey variance mass (5th–95th percentile band across subsets)
    ax.fill_between(x_plot, lo, hi, color="grey", alpha=0.35, label="Variance band (5–95%ile)")

    # Individual model fits (a lot more curves for texture)
    n_show = len(preds)
    cmap = plt.cm.Blues if "Degree 1" in title else (plt.cm.Greens if "Degree 5" in title else plt.cm.Purples)
    colors = cmap(np.linspace(0.25, 0.95, n_show))
    colors[:, -1] = 0.035  # low alpha keeps the plot readable
    for i in range(n_show):
        ax.plot(x_plot, preds[i], color=colors[i], linewidth=0.9)

    # Ground truth
    ax.plot(x_plot, y_true_plot, color="crimson", linewidth=2, label="Ground truth $f(x)$")
    ax.scatter(x_all, y_all, color="steelblue", s=40, zorder=5, label="Candidate sample points")

    # Average prediction (bias proxy)
    ax.plot(x_plot, mean_pred, color="black", linewidth=2.2,
            linestyle="--", label="Mean Prediction $\\bar{g}(x)$")

    # Uncertainty around the mean (std across subsets)
    std_pred = preds.std(axis=0)
    ax.plot(x_plot, mean_pred + std_pred, color="black", linewidth=1.2, alpha=0.35, label="Mean +/- 1 std")
    ax.plot(x_plot, mean_pred - std_pred, color="black", linewidth=1.2, alpha=0.35)

    ax.set_xlim(x_plot[0], x_plot[-1])
    ax.set_ylim(y_lim)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8)

    # Bias / variance annotation below plot
    ax.text(0.5, -0.12,
            f"Bias²  = {bias_val:.4f}    Variance = {var_val:.4f}    "
            f"Bias² + Var = {bias_val + var_val:.4f}",
            transform=ax.transAxes,
            ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"))

fig.text(0.5, 0.01,
         f"Each model trained on {N_TRAIN} of {N_TOTAL} total points  "
         f"($\\binom{{{N_TOTAL}}}{{{N_TRAIN}}} = 3003$ possible combinations; "
         f"{N_SUBSETS} randomly sampled here)\n"
         f"Ground truth $f(x)$ is a Degree-5 polynomial",
         ha="center", va="bottom", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"))
plt.savefig("outputs/bias_var_viz.png", dpi=150, bbox_inches="tight")
plt.show()
