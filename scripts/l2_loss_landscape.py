"""
l2_loss_landscape.py

Visualise the L2 (mean-squared-error) loss landscape for a simple linear model
  ŷ = w1*x + w0

Three panels:
  Left   – generated data with the optimal fitted line
  Centre – filled-contour loss landscape over (w0, w1) with gradient-descent paths
  Right  – 3-D surface of the same landscape from a tilted angle
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – required for 3-D projection

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------------------------------
TRUE_W0 =  2.0   # intercept
TRUE_W1 =  1.5   # slope
NOISE   =  1.2

rng = np.random.default_rng(42)
x   = np.linspace(-4, 4, 30)
y   = TRUE_W1 * x + TRUE_W0 + rng.normal(0, NOISE, size=len(x))

# ---------------------------------------------------------------------------
# L2 loss function
# ---------------------------------------------------------------------------

def l2_loss(w0: float, w1: float) -> float:
    """MSE = (1/n) sum (y_i - (w1*x_i + w0))^2"""
    residuals = y - (w1 * x + w0)
    return float(np.mean(residuals ** 2))


def l2_loss_grid(W0: np.ndarray, W1: np.ndarray) -> np.ndarray:
    """Vectorised loss over meshgrid arrays."""
    # W0, W1 have shape (n, n); x/y are 1-D (n_data,)
    # Broadcast: residuals shape = (n_grid, n_grid, n_data)
    residuals = y[None, None, :] - (W1[:, :, None] * x[None, None, :] + W0[:, :, None])
    return np.mean(residuals ** 2, axis=-1)


# ---------------------------------------------------------------------------
# Analytical optimum (normal equations for 1-D linear regression)
# ---------------------------------------------------------------------------
x_mean, y_mean = x.mean(), y.mean()
w1_opt = np.dot(x - x_mean, y - y_mean) / np.dot(x - x_mean, x - x_mean)
w0_opt = y_mean - w1_opt * x_mean
loss_opt = l2_loss(w0_opt, w1_opt)

# ---------------------------------------------------------------------------
# Loss landscape grid
# ---------------------------------------------------------------------------
W0_RANGE = (w0_opt - 5, w0_opt + 5)
W1_RANGE = (w1_opt - 3, w1_opt + 3)
N_GRID   = 200

w0_vals = np.linspace(*W0_RANGE, N_GRID)
w1_vals = np.linspace(*W1_RANGE, N_GRID)
W0_grid, W1_grid = np.meshgrid(w0_vals, w1_vals)
L_grid  = l2_loss_grid(W0_grid, W1_grid)

# ---------------------------------------------------------------------------
# Gradient descent (closed-form gradient of MSE)
# ---------------------------------------------------------------------------

def grad_l2(w0: float, w1: float) -> tuple[float, float]:
    """(dL/dw0, dL/dw1)"""
    residuals = y - (w1 * x + w0)
    dw0 = -2.0 * residuals.mean()
    dw1 = -2.0 * (residuals * x).mean()
    return dw0, dw1


def gradient_descent(w0_init, w1_init, lr=0.05, n_steps=120):
    path = [(w0_init, w1_init)]
    w0, w1 = w0_init, w1_init
    for _ in range(n_steps):
        dw0, dw1 = grad_l2(w0, w1)
        w0 -= lr * dw0
        w1 -= lr * dw1
        path.append((w0, w1))
    return np.array(path)


STARTS = [
    (w0_opt - 4.5, w1_opt - 2.5),
    (w0_opt + 4.0, w1_opt + 2.5),
    (w0_opt - 3.5, w1_opt + 2.8),
    (w0_opt + 3.5, w1_opt - 2.2),
]

paths = [gradient_descent(*s) for s in STARTS]
PATH_COLORS = ["#E87B4C", "#4C9BE8", "#7BE87B", "#E84CE8"]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(8, 7))
fig.suptitle(
    r"L2 Loss Landscape  —  $\hat{y} = w_1 x + w_0$,  "
    r"$\mathcal{L}(w_0, w_1) = \frac{1}{n}\sum_i\,(y_i - \hat{y}_i)^2$",
    fontsize=14, fontweight="bold", y=0.99,
)

# ax_data    = fig.add_subplot(gs[0])
# ax_contour = fig.add_subplot(gs[1])
ax_3d = fig.add_subplot(111, projection="3d")

# # ── Panel 1: data + optimal fit ───────────────────────────────────────────
# ax_data.scatter(x, y, color="steelblue", s=40, zorder=4, label="Observed data")
# x_line = np.array([x.min(), x.max()])
# ax_data.plot(x_line, w1_opt * x_line + w0_opt,
#              color="crimson", linewidth=2,
#              label=f"Optimal fit\n$w_0={w0_opt:.2f}$, $w_1={w1_opt:.2f}$")
# ax_data.plot(x_line, TRUE_W1 * x_line + TRUE_W0,
#              color="grey", linewidth=1.5, linestyle="--",
#              label=f"True line\n$w_0={TRUE_W0}$, $w_1={TRUE_W1}$")
# ax_data.set_title("Data & Fitted Line", fontsize=12)
# ax_data.set_xlabel("x")
# ax_data.set_ylabel("y")
# ax_data.legend(fontsize=8)
# ax_data.annotate(
#     f"MSE* = {loss_opt:.3f}",
#     xy=(0.97, 0.05), xycoords="axes fraction",
#     ha="right", fontsize=9,
#     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"),
# )

# # ── Panel 2: contour loss landscape ──────────────────────────────────────
# # Use log-scale levels so the bowl's steep walls and flat floor are both visible
# L_log   = np.log1p(L_grid - loss_opt)          # shift so minimum ≈ 0
# n_levels = 30
# levels   = np.linspace(L_log.min(), L_log.max(), n_levels)
#
# cf = ax_contour.contourf(W0_grid, W1_grid, L_log,
#                           levels=levels, cmap="YlOrRd_r", alpha=0.85)
# ax_contour.contour(W0_grid, W1_grid, L_log,
#                    levels=levels[::3], colors="white", linewidths=0.4, alpha=0.3)
#
# cbar = fig.colorbar(cf, ax=ax_contour, fraction=0.046, pad=0.04)
# cbar.set_label(r"$\log(1 + \mathcal{L} - \mathcal{L}^*)$", fontsize=9)
# cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
#
# # Gradient descent paths
# for path, color in zip(paths, PATH_COLORS):
#     ax_contour.plot(path[:, 0], path[:, 1],
#                     color=color, linewidth=1.5, alpha=0.85, zorder=5)
#     ax_contour.plot(*path[0], "o", color=color, markersize=7,
#                     markeredgecolor="white", markeredgewidth=0.8, zorder=6)
#     ax_contour.annotate("start", xy=path[0], xytext=(path[0][0] + 0.15, path[0][1] + 0.10),
#                         color=color, fontsize=7, zorder=7)
#
# # Optimum
# ax_contour.plot(w0_opt, w1_opt, "*", color="white", markersize=14,
#                 markeredgecolor="black", markeredgewidth=0.7, zorder=8,
#                 label=f"Minimum  ($w_0^*={w0_opt:.2f}$, $w_1^*={w1_opt:.2f}$)")
# ax_contour.set_xlabel("$w_0$  (intercept)", fontsize=10)
# ax_contour.set_ylabel("$w_1$  (slope)", fontsize=10)
# ax_contour.set_title("Loss Landscape + Gradient Descent", fontsize=12)
# ax_contour.legend(fontsize=8, loc="upper right",
#                   facecolor="white", edgecolor="grey", framealpha=0.8)

# ── Panel 3: 3-D surface ─────────────────────────────────────────────────
# Subsample for speed
step = 4
W0_s = W0_grid[::step, ::step]
W1_s = W1_grid[::step, ::step]
L_s  = L_grid[::step, ::step]

surf = ax_3d.plot_surface(
    W0_s, W1_s, L_s,
    cmap="YlOrRd_r",
    linewidth=0, antialiased=True, alpha=0.88,
)

# Mark the minimum
ax_3d.scatter([w0_opt], [w1_opt], [loss_opt],
              color="white", s=60, edgecolors="black", linewidths=0.8,
              zorder=10, label="Minimum")

ax_3d.set_xlabel("$w_0$", labelpad=6, fontsize=9)
ax_3d.set_ylabel("$w_1$", labelpad=6, fontsize=9)
ax_3d.set_zlabel(r"$\mathcal{L}(w_0, w_1)$", labelpad=6, fontsize=9)
ax_3d.set_title("3-D View", fontsize=12)
ax_3d.view_init(elev=28, azim=-55)
ax_3d.tick_params(labelsize=7)
ax_3d.legend(fontsize=8, loc="upper right")

# ---------------------------------------------------------------------------
# Save figure 1
# ---------------------------------------------------------------------------
stamp    = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
out_path = OUTPUT_DIR / f"{stamp}_l2_loss_landscape.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to: {out_path}")

# ===========================================================================
# Figure 2 – loss landscape for a 3-weight model: y = w0 + w1*x + w2*x²
#
# The full landscape lives in 3-D weight space (a 3-D paraboloid).
# We visualise it as three pairwise 2-D slices, each holding the third
# weight fixed at its optimum.
# ===========================================================================

# ── Optimal weights via normal equations ────────────────────────────────────
X_poly = np.column_stack([np.ones_like(x), x, x ** 2])   # (n, 3)
w_opt3, _, _, _ = np.linalg.lstsq(X_poly, y, rcond=None)  # [w0*, w1*, w2*]
w0_opt3, w1_opt3, w2_opt3 = w_opt3


def l2_loss3(w0, w1, w2):
    return float(np.mean((y - (w0 + w1 * x + w2 * x ** 2)) ** 2))


loss_opt3 = l2_loss3(w0_opt3, w1_opt3, w2_opt3)

# ── Grids for each pair of weights ──────────────────────────────────────────
N3    = 120
SLACK = 3.0   # half-range around each optimum

def make_slice(wa_opt, wb_opt, fix_val, fix_axis, n=N3):
    """Return meshgrid + loss surface for a (wa, wb) slice, fixing the third weight."""
    wa = np.linspace(wa_opt - SLACK, wa_opt + SLACK, n)
    wb = np.linspace(wb_opt - SLACK, wb_opt + SLACK, n)
    WA, WB = np.meshgrid(wa, wb)
    L = np.zeros_like(WA)
    for i in range(n):
        for j in range(n):
            args = [None, None, None]
            args[fix_axis] = fix_val
            free = [k for k in range(3) if k != fix_axis]
            args[free[0]] = WA[i, j]
            args[free[1]] = WB[i, j]
            L[i, j] = l2_loss3(*args)
    return WA, WB, L

# Slice 1: (w0, w1), fix w2 = w2_opt3
WA_01, WB_01, L_01 = make_slice(w0_opt3, w1_opt3, w2_opt3, fix_axis=2)
# Slice 2: (w0, w2), fix w1 = w1_opt3
WA_02, WB_02, L_02 = make_slice(w0_opt3, w2_opt3, w1_opt3, fix_axis=1)
# Slice 3: (w1, w2), fix w0 = w0_opt3
WA_12, WB_12, L_12 = make_slice(w1_opt3, w2_opt3, w0_opt3, fix_axis=0)

# ── Figure ───────────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(20, 7))
fig2.suptitle(
    r"L2 Loss Landscape — 3-weight model  $\hat{y} = w_0 + w_1 x + w_2 x^2$"
    "\n"
    r"Each panel: 2-D slice through weight space, third weight fixed at $w^*$",
    fontsize=13, fontweight="bold", y=1.005,
)

slices = [
    (WA_01, WB_01, L_01, "$w_0$", "$w_1$",
     f"$w_2$ fixed at {w2_opt3:.2f}", w0_opt3, w1_opt3),
    (WA_02, WB_02, L_02, "$w_0$", "$w_2$",
     f"$w_1$ fixed at {w1_opt3:.2f}", w0_opt3, w2_opt3),
    (WA_12, WB_12, L_12, "$w_1$", "$w_2$",
     f"$w_0$ fixed at {w0_opt3:.2f}", w1_opt3, w2_opt3),
]

for k, (WA, WB, L, xlabel, ylabel, fixed_label, wa_star, wb_star) in enumerate(slices):
    ax = fig2.add_subplot(1, 3, k + 1, projection="3d")

    step3 = max(1, N3 // 50)
    surf = ax.plot_surface(
        WA[::step3, ::step3], WB[::step3, ::step3], L[::step3, ::step3],
        cmap="YlOrRd_r", linewidth=0, antialiased=True, alpha=0.88,
    )

    loss_at_star = l2_loss3(
        w0_opt3 if xlabel == "$w_0$" or ylabel == "$w_0$" else w0_opt3,
        w1_opt3 if xlabel == "$w_1$" or ylabel == "$w_1$" else w1_opt3,
        w2_opt3 if xlabel == "$w_2$" or ylabel == "$w_2$" else w2_opt3,
    )
    ax.scatter([wa_star], [wb_star], [loss_at_star],
               color="white", s=60, edgecolors="black", linewidths=0.8, zorder=10)

    ax.set_xlabel(xlabel, labelpad=6, fontsize=10)
    ax.set_ylabel(ylabel, labelpad=6, fontsize=10)
    ax.set_zlabel(r"$\mathcal{L}$", labelpad=6, fontsize=10)
    ax.set_title(f"Slice: {xlabel} vs {ylabel}\n({fixed_label})", fontsize=11)
    ax.view_init(elev=28, azim=-55)
    ax.tick_params(labelsize=7)

fig2.tight_layout()
out_path2 = OUTPUT_DIR / f"{stamp}_l2_loss_landscape_3weights.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved to: {out_path2}")

# ===========================================================================
# Figure 3 – L2 loss as a function of the error e = y - ŷ
# ===========================================================================
e_range = np.linspace(-4, 4, 400)
L_of_e  = e_range ** 2

fig3, ax_resid = plt.subplots(figsize=(7, 5))
fig3.suptitle(
    r"L2 Loss Landscape  —  $\mathcal{L}(y, \hat{y}) = \frac{1}{n}\sum_i\,(y_i - \hat{y}_i)^2$",
    fontsize=13, fontweight="bold",
)

# ── Loss curve with MSE reference line ──────────────────────────────────────
residuals    = y - (w1_opt * x + w0_opt)
mean_e       = residuals.mean()           # E[e]
f_of_mean_e  = mean_e ** 2               # f(E[e]) = (E[e])²  — left side of Jensen
E_of_f_e     = loss_opt                  # E[f(e)] = E[e²] = MSE*  — right side
jensen_gap   = E_of_f_e - f_of_mean_e    # = Var(e)

ax_resid.plot(e_range, L_of_e, color="crimson", linewidth=2,
              label=r"$\mathcal{L}(e) = e^2$")
ax_resid.axhline(E_of_f_e, color="grey", linewidth=1.2, linestyle=":",
                 label=rf"$\mathbb{{E}}[e^2]$ = MSE* = {E_of_f_e:.3f}")

# ── Jensen's inequality ───────────────────────────────────────────────────────
# Mark E[e] on the x-axis and f(E[e]) on the curve
ax_resid.axvline(mean_e, color="steelblue", linewidth=1.2, linestyle="--",
                 label=rf"$\mathbb{{E}}[e]$ = {mean_e:.3f}")
ax_resid.scatter([mean_e], [f_of_mean_e], color="steelblue", s=60, zorder=6,
                 label=rf"$f(\mathbb{{E}}[e])$ = $(\mathbb{{E}}[e])^2$ = {f_of_mean_e:.4f}")

# Vertical bracket showing the Jensen gap
ax_resid.annotate(
    "", xy=(mean_e + 0.15, E_of_f_e), xytext=(mean_e + 0.15, f_of_mean_e),
    arrowprops=dict(arrowstyle="<->", color="darkorange", lw=1.5),
)
ax_resid.text(
    mean_e + 0.22, (f_of_mean_e + E_of_f_e) / 2,
    rf"Jensen gap$= \mathrm{{Var}}(e) = {jensen_gap:.3f}$",
    fontsize=8, color="darkorange", va="center",
)

# Inequality label
ax_resid.text(
    0.03, 0.93,
    r"Jensen's inequality: $f\!\left(\mathbb{E}[e]\right) \leq \mathbb{E}\!\left[f(e)\right]$"
    "\n"
    r"i.e. $\left(\mathbb{E}[e]\right)^2 \leq \mathbb{E}[e^2]$",
    transform=ax_resid.transAxes, fontsize=9, va="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="grey"),
)

ax_resid.set_xlabel("Error  $e = y - \\hat{y}$", fontsize=11)
ax_resid.set_ylabel("$e^2$", fontsize=11)
ax_resid.legend(fontsize=9)
ax_resid.set_xlim(e_range[0], e_range[-1])

fig3.tight_layout()
out_path3 = OUTPUT_DIR / f"{stamp}_l2_loss_vs_error.png"
plt.savefig(out_path3, dpi=150, bbox_inches="tight")
print(f"Saved to: {out_path3}")

plt.show()

# ===========================================================================
# Figure 4 – Jensen's inequality: chord vs. curve (reference-image style)
#   f(x) = x²,  two points x1 < x2,  convex combination t*x1 + (1-t)*x2
# ===========================================================================
theta = 0.35              # convex-combination weight (θ)
x1_j, x2_j = -1.5, 2.8   # pick so parabola has a visible minimum between them
f4 = lambda v: v ** 2

xt   = theta * x1_j + (1 - theta) * x2_j        # θx1 + (1-θ)x2
f_xt = f4(xt)                                    # f(θx1 + (1-θ)x2)  — ON curve
c_xt = theta * f4(x1_j) + (1 - theta) * f4(x2_j)  # θf(x1)+(1-θ)f(x2) — ON chord

# Extend the chord slightly beyond x1 and x2 for the line
slope_chord = (f4(x2_j) - f4(x1_j)) / (x2_j - x1_j)
x_chord_lo, x_chord_hi = x1_j - 0.6, x2_j + 0.5
chord_line_x = np.array([x_chord_lo, x_chord_hi])
chord_line_y = f4(x1_j) + slope_chord * (chord_line_x - x1_j)

e4 = np.linspace(-2.2, 3.4, 500)

fig4, ax4 = plt.subplots(figsize=(7, 5.5))
fig4.suptitle(
    r"L2 Loss Landscape  —  $\mathcal{L}(y,\hat{y}) = \frac{1}{n}\sum_i\,(y_i - \hat{y}_i)^2$",
    fontsize=13, fontweight="bold", y=0.98,
)

# ── Minimal axes style (arrows, no box) ──────────────────────────────────
for spine in ["top", "right"]:
    ax4.spines[spine].set_visible(False)
ax4.spines["bottom"].set_position("zero")
ax4.spines["left"].set_position("zero")
ax4.spines["bottom"].set_linewidth(0.8)
ax4.spines["left"].set_linewidth(0.8)
ax4.set_xticks([])
ax4.set_yticks([])

# Arrow heads on axes
ax4.annotate("", xy=(e4[-1] + 0.15, 0), xytext=(e4[0] - 0.1, 0),
             arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))
ax4.annotate("", xy=(0, f4(x2_j) * 1.12), xytext=(0, -0.5),
             arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))

# ── Curve ────────────────────────────────────────────────────────────────
ax4.plot(e4, f4(e4), color="black", linewidth=2, zorder=3)

# ── Chord (magenta, extended) ─────────────────────────────────────────────
ax4.plot(chord_line_x, chord_line_y, color="magenta", linewidth=1.8, zorder=2)

# ── Dashed verticals at x1, xt, x2 ───────────────────────────────────────
grey = dict(color="grey", linewidth=0.9, linestyle="--")
for xi, top_y in [(x1_j, f4(x1_j)), (xt, c_xt), (x2_j, f4(x2_j))]:
    ax4.plot([xi, xi], [0, top_y], **grey, zorder=1)

# ── Dashed horizontals for the two Jensen values ──────────────────────────
ax4.plot([e4[0] - 0.05, xt], [c_xt, c_xt], **grey, zorder=1)
ax4.plot([e4[0] - 0.05, xt], [f_xt, f_xt], **grey, zorder=1)

# ── Dots at the four key points ───────────────────────────────────────────
ax4.scatter([x1_j, x2_j], [f4(x1_j), f4(x2_j)],
            color="black", s=40, zorder=5)                  # curve endpoints
ax4.scatter([xt], [c_xt], color="magenta", s=55, zorder=6)  # chord point
ax4.scatter([xt], [f_xt], color="black",   s=55, zorder=6)  # curve point

# ── x-axis labels ─────────────────────────────────────────────────────────
for xi, lbl in [(x1_j, r"$x_1$"), (xt, r"$\theta x_1\!+\!(1\!-\!\theta)x_2$"), (x2_j, r"$x_2$")]:
    ax4.text(xi, -0.55, lbl, ha="center", va="top", fontsize=10, color="black")

# ── y-axis labels ─────────────────────────────────────────────────────────
x_lbl = e4[0] - 0.12
ax4.text(x_lbl, c_xt, r"$\theta f(x_1)+(1-\theta)f(x_2)$", ha="right", va="center", fontsize=9)
ax4.text(x_lbl, f_xt, r"$f(\theta x_1+(1-\theta)x_2)$",   ha="right", va="center", fontsize=9)

ax4.set_xlim(e4[0] - 2.2, e4[-1] + 0.4)
ax4.set_ylim(-0.8, f4(x2_j) * 1.15)

fig4.tight_layout()
out_path4 = OUTPUT_DIR / f"{stamp}_l2_jensen_chord.png"
plt.savefig(out_path4, dpi=150, bbox_inches="tight")
print(f"Saved to: {out_path4}")

plt.show()
