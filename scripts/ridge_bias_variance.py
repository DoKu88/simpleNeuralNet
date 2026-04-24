"""
ridge_bias_variance.py

GIF: sweeps λ from near-0 to large, showing how L2 ridge regression
trades off bias and variance across repeated noisy dataset realisations.

A degree-9 polynomial model is fit to 15 noisy samples of a smooth
ground truth. Small λ → overfit (high variance, low bias).
Large λ → underfit (low variance, high bias).

Layout
------
Left (full height):  60 ridge fits from independent noise realisations (faint)
                     + mean prediction + ground truth + one example scatter
Top right:           Bias²(λ), Variance(λ), Bias²+Variance vs λ (log scale)
Bottom right:        Current Bias² vs Variance bar chart
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_COL  = "#21262d"
TEXT_COL  = "#e6edf3"
BIAS_COL  = "#e74c3c"
VAR_COL   = "#3498db"
TOT_COL   = "#2ecc71"
TRUTH_COL = "#f1c40f"
MEAN_COL  = "#ff6b6b"

# ── Ground truth + data ──────────────────────────────────────────────────────────
def f_true(x):
    return np.sin(2.5 * x) + 0.4 * np.cos(5 * x)


NOISE_STD  = 1.0
N_TRAIN    = 15
N_DATASETS = 60
DEGREE     = 13         # 14 features, 15 samples → near-interpolating, high variance at small λ

rng = np.random.default_rng(42)

x_train     = np.linspace(0, np.pi, N_TRAIN)
x_plot      = np.linspace(0, np.pi, 300)
y_true_plot = f_true(x_plot)

# ── Design matrices ───────────────────────────────────────────────────────────────
def poly_matrix(x, degree):
    return np.column_stack([np.ones_like(x)] + [x ** d for d in range(1, degree + 1)])

X_train_raw = poly_matrix(x_train, DEGREE)
X_plot_raw  = poly_matrix(x_plot,  DEGREE)

# Standardise non-intercept columns for numerical stability
feat_mean = X_train_raw[:, 1:].mean(axis=0)
feat_std  = X_train_raw[:, 1:].std(axis=0) + 1e-8

X_train = np.hstack([X_train_raw[:, :1],
                     (X_train_raw[:, 1:] - feat_mean) / feat_std])
X_plot  = np.hstack([X_plot_raw[:, :1],
                     (X_plot_raw[:, 1:] - feat_mean) / feat_std])

# ── Noisy dataset realisations ────────────────────────────────────────────────────
y_clean    = f_true(x_train)
Y_datasets = y_clean + rng.normal(0, NOISE_STD, (N_DATASETS, N_TRAIN))

# ── Ridge helper ─────────────────────────────────────────────────────────────────
n_feat = DEGREE + 1
XtX    = X_train.T @ X_train
I_p    = np.diag([0.0] + [1.0] * DEGREE)   # intercept column left unpenalised


def ridge_preds_all(lam):
    """Returns (N_DATASETS, N_plot) predictions for all datasets at once."""
    Xty = X_train.T @ Y_datasets.T                    # (n_feat, N_DATASETS)
    W   = np.linalg.solve(XtX + lam * I_p, Xty)      # (n_feat, N_DATASETS)
    return (X_plot @ W).T                              # (N_DATASETS, N_plot)


# ── Precompute animation data ─────────────────────────────────────────────────────
N_FRAMES     = 80
lambdas_anim = np.logspace(-3, 2, N_FRAMES)           # 0.001 → 100

print("Precomputing predictions for animation frames...")
all_preds = np.stack([ridge_preds_all(lam) for lam in lambdas_anim])
# shape: (N_FRAMES, N_DATASETS, N_plot)
print("Done.")

mean_preds = all_preds.mean(axis=1)                                  # (N_FRAMES, N_plot)
bias2_anim = ((mean_preds - y_true_plot) ** 2).mean(axis=1)         # (N_FRAMES,)
var_anim   = all_preds.var(axis=1).mean(axis=1)                     # (N_FRAMES,)

# Dense curves for smooth right-panel backgrounds
N_DENSE       = 300
lambdas_dense = np.logspace(-3, 2, N_DENSE)
bias2_dense   = np.zeros(N_DENSE)
var_dense     = np.zeros(N_DENSE)
for i, lam in enumerate(lambdas_dense):
    pd = ridge_preds_all(lam)
    bias2_dense[i] = ((pd.mean(axis=0) - y_true_plot) ** 2).mean()
    var_dense[i]   = pd.var(axis=0).mean()
total_dense = bias2_dense + var_dense

# ── Figure ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 8), facecolor=DARK_BG)
gs  = fig.add_gridspec(2, 2,
                        width_ratios=[1.5, 1],
                        hspace=0.42, wspace=0.32,
                        left=0.07, right=0.97, top=0.87, bottom=0.09)

ax_left = fig.add_subplot(gs[:, 0])
ax_rt   = fig.add_subplot(gs[0, 1])
ax_rb   = fig.add_subplot(gs[1, 1])

for ax in [ax_left, ax_rt, ax_rb]:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#30363d")

# ── Titles ───────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.962,
    r"Ridge Regression Bias–Variance Tradeoff: "
    r"$\hat{w}_\lambda = (X^\top X + \lambda I)^{-1} X^\top y$",
    ha="center", va="center", fontsize=13, color=TEXT_COL, fontweight="bold",
)
lambda_title = fig.text(
    0.5, 0.924,
    rf"$\lambda = {lambdas_anim[0]:.3f}$",
    ha="center", va="center", fontsize=13, color=TEXT_COL, fontweight="bold",
)

# ── Left panel ────────────────────────────────────────────────────────────────────
ax_left.plot(x_plot, y_true_plot,
             color=TRUTH_COL, lw=2.5, ls="--", zorder=10,
             label="Ground truth $f(x)$")

fit_lines = []
for j in range(N_DATASETS):
    ln, = ax_left.plot(x_plot, all_preds[0, j],
                       color="#8b949e", lw=0.7, alpha=0.20, zorder=2)
    fit_lines.append(ln)

mean_line, = ax_left.plot(x_plot, mean_preds[0],
                           color=MEAN_COL, lw=2.5, zorder=8,
                           label=r"Mean prediction $\bar{\hat{y}}$")

ax_left.scatter(x_train, Y_datasets[0],
                s=35, color="#4a9edd", alpha=0.65, zorder=9,
                label="Sample data (one realisation)")

y_lo = np.percentile(all_preds, 0.5)
y_hi = np.percentile(all_preds, 99.5)
pad  = 0.18 * (y_hi - y_lo)
ax_left.set_xlim(x_plot[0], x_plot[-1])
ax_left.set_ylim(y_lo - pad, y_hi + pad)
ax_left.set_xlabel(r"$x$", color=TEXT_COL, fontsize=12)
ax_left.set_ylabel(r"$y$", color=TEXT_COL, fontsize=12)
ax_left.set_title(
    r"Ridge Fits from 60 Noise Realisations — spread = variance, offset = bias",
    color=TEXT_COL, fontsize=10, pad=8,
)
ax_left.legend(fontsize=9, facecolor=PANEL_BG, edgecolor="#30363d",
               labelcolor=TEXT_COL, loc="lower left")
ax_left.grid(True, color=GRID_COL, alpha=0.3)

# ── Top-right: Bias²/Variance/Total curves ────────────────────────────────────────
ax_rt.semilogx(lambdas_dense, bias2_dense, color=BIAS_COL, lw=2.0,
               label=r"Bias$^2(\lambda)$")
ax_rt.semilogx(lambdas_dense, var_dense,   color=VAR_COL,  lw=2.0,
               label=r"Variance$(\lambda)$")
ax_rt.semilogx(lambdas_dense, total_dense, color=TOT_COL,  lw=2.0, ls="--",
               label=r"Bias$^2 +$ Variance")

cursor_vline = ax_rt.axvline(lambdas_anim[0], color="white", lw=1.5, ls="--",
                              alpha=0.85, zorder=5)
bias_dot,  = ax_rt.plot([lambdas_anim[0]], [bias2_anim[0]],
                         "o", color=BIAS_COL, ms=8,
                         markeredgecolor="white", markeredgewidth=0.8, zorder=6)
var_dot,   = ax_rt.plot([lambdas_anim[0]], [var_anim[0]],
                         "o", color=VAR_COL,  ms=8,
                         markeredgecolor="white", markeredgewidth=0.8, zorder=6)
total_dot, = ax_rt.plot([lambdas_anim[0]], [bias2_anim[0] + var_anim[0]],
                         "o", color=TOT_COL, ms=8,
                         markeredgecolor="white", markeredgewidth=0.8, zorder=6)

ax_rt.set_xlabel(r"$\lambda$  (log scale)", color=TEXT_COL, fontsize=10)
ax_rt.set_ylabel("Error", color=TEXT_COL, fontsize=10)
ax_rt.set_title(r"Bias$^2$ and Variance vs $\lambda$",
                color=TEXT_COL, fontsize=10, pad=8)
ax_rt.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#30363d",
             labelcolor=TEXT_COL)
ax_rt.grid(True, color=GRID_COL, alpha=0.5)

# ── Bottom-right: current decomposition bar ───────────────────────────────────────
bar_max = total_dense.max() * 1.15
tot0    = bias2_anim[0] + var_anim[0]
bars = ax_rb.bar(
    ["Bias²", "Variance", "Total"],
    [bias2_anim[0], var_anim[0], tot0],
    color=[BIAS_COL, VAR_COL, TOT_COL],
    edgecolor="white", linewidth=0.8, width=0.5,
)
ax_rb.set_ylim(0, bar_max)
ax_rb.set_ylabel("Error magnitude", color=TEXT_COL, fontsize=10)
ax_rb.set_title("Current decomposition", color=TEXT_COL, fontsize=10, pad=8)
ax_rb.grid(True, color=GRID_COL, alpha=0.4, axis="y")

bias_bar_txt = ax_rb.text(
    0, bias2_anim[0] + 0.02 * bar_max,
    f"{bias2_anim[0]:.4f}",
    ha="center", va="bottom", color=TEXT_COL, fontsize=9, fontweight="bold",
)
var_bar_txt = ax_rb.text(
    1, var_anim[0] + 0.02 * bar_max,
    f"{var_anim[0]:.4f}",
    ha="center", va="bottom", color=TEXT_COL, fontsize=9, fontweight="bold",
)
tot_bar_txt = ax_rb.text(
    2, tot0 + 0.02 * bar_max,
    f"{tot0:.4f}",
    ha="center", va="bottom", color=TEXT_COL, fontsize=9, fontweight="bold",
)


# ── Update function ───────────────────────────────────────────────────────────────
def update(frame):
    lam = lambdas_anim[frame]
    b2  = bias2_anim[frame]
    v   = var_anim[frame]

    for j, ln in enumerate(fit_lines):
        ln.set_ydata(all_preds[frame, j])
    mean_line.set_ydata(mean_preds[frame])

    cursor_vline.set_xdata([lam, lam])
    bias_dot.set_data([lam], [b2])
    var_dot.set_data([lam], [v])
    total_dot.set_data([lam], [b2 + v])

    tot = b2 + v
    bars[0].set_height(b2)
    bars[1].set_height(v)
    bars[2].set_height(tot)
    bias_bar_txt.set_position((0, b2  + 0.02 * bar_max))
    bias_bar_txt.set_text(f"{b2:.4f}")
    var_bar_txt.set_position((1, v   + 0.02 * bar_max))
    var_bar_txt.set_text(f"{v:.4f}")
    tot_bar_txt.set_position((2, tot + 0.02 * bar_max))
    tot_bar_txt.set_text(f"{tot:.4f}")

    lambda_title.set_text(rf"$\lambda = {lam:.3f}$")

    return [*fit_lines, mean_line, cursor_vline,
            bias_dot, var_dot, total_dot,
            bars[0], bars[1], bars[2],
            bias_bar_txt, var_bar_txt, tot_bar_txt, lambda_title]


anim = animation.FuncAnimation(
    fig, update, frames=N_FRAMES, interval=100, blit=False,
)

stamp    = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
out_path = OUTPUT_DIR / f"{stamp}_ridge_bias_variance.gif"
anim.save(str(out_path), writer="pillow", fps=10, dpi=110)
print(f"Saved to: {out_path}")
plt.show()
