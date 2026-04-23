"""
ridge_lambda_animation.py

GIF: sweeps λ from near-0 to large, showing

    w_ridge(λ) = Q · diag( λ_i / (λ_i + λ) ) · Q^T · w_opt

Three visible consequences:
  λ → 0:   w_ridge → w_opt          (factors → 1)
  λ → ∞:   w_ridge → 0             (factors → 0)
  General:  smaller λ_i shrunk more  (differential shrinkage)
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Synthetic data ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)
n   = 80

x1 = rng.normal(0, 1, n)
x2 = 0.90 * x1 + rng.normal(0, 0.44, n)
X  = np.column_stack([x1, x2])
y  = X @ np.array([1.5, 0.8]) + rng.normal(0, 0.6, n)
X -= X.mean(axis=0)
y -= y.mean()

# ── Eigendecomposition ─────────────────────────────────────────────────────────
XtX = X.T @ X
Xty = X.T @ y
eigenvalues, Q = np.linalg.eigh(XtX)   # ascending: [λ_small, λ_large]
w_ols = np.linalg.solve(XtX, Xty)


def ridge_w(lam):
    if lam <= 0:
        return w_ols.copy()
    return np.linalg.solve(XtX + lam * np.eye(2), Xty)


# ── Static background data ─────────────────────────────────────────────────────
lambdas_full = np.logspace(-2, 2.7, 500)
w_path       = np.array([ridge_w(l) for l in lambdas_full])
shrink_full  = np.stack([ev / (ev + lambdas_full) for ev in eigenvalues], axis=1)

# OLS loss landscape for weight-space panel
pad  = 2.5
w1g  = np.linspace(w_ols[0] - pad, w_ols[0] + pad, 180)
w2g  = np.linspace(w_ols[1] - pad, w_ols[1] + pad, 180)
W1G, W2G = np.meshgrid(w1g, w2g)
Wf   = np.stack([W1G.ravel(), W2G.ravel()], axis=1)
LOSS = np.sum((y[:, None] - X @ Wf.T) ** 2, axis=0).reshape(W1G.shape)
loss_ols       = float(np.sum((y - X @ w_ols) ** 2))
contour_levels = loss_ols + np.array([1, 4, 12, 30, 65, 120])

# x1-axis data for the regression panel (x2 held at mean = 0)
x1_vals = X[:, 0]
x1_plot = np.linspace(x1_vals.min(), x1_vals.max(), 200)

# ── Animation frames ───────────────────────────────────────────────────────────
N_FRAMES     = 100
lambdas_anim = np.logspace(-2, 2.7, N_FRAMES)

EI_COLORS = ["#e74c3c", "#3498db"]   # small eigenvalue red, large blue
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_COL  = "#21262d"
TEXT_COL  = "#e6edf3"

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9), facecolor=DARK_BG)
gs  = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.30,
                        left=0.05, right=0.97, top=0.865, bottom=0.08,
                        width_ratios=[1.25, 1])
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1])

for ax in [ax00, ax01, ax10, ax11]:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#30363d")

# ── Title: equations on top line, animated λ on the line below ────────────────
fig.text(
    0.5, 0.978,
    r"$\mathcal{L}(w)=\|y-Xw\|^2+\lambda\|w\|^2$"
    r"$\qquad$"
    r"$w_{\rm ridge}=Q\,\mathrm{diag}\!\left(\frac{\lambda_i}{\lambda_i+\lambda}"
    r"\right)Q^\top w_{\rm OLS}$",
    ha="center", va="center", fontsize=14, color=TEXT_COL, fontweight="bold",
)
super_title = fig.text(
    0.5, 0.940,
    r"$\lambda=0.01$",
    ha="center", va="center", fontsize=14, color=TEXT_COL, fontweight="bold",
)

# ── [0,0]: Shrinkage factor bars ───────────────────────────────────────────────
ax00.set_xlim(-0.6, 1.6)
ax00.set_ylim(0, 1.22)
ax00.set_xticks([0, 1])
ax00.set_xticklabels(
    [rf"$\lambda_{{\rm small}}$" + f"\n= {eigenvalues[0]:.1f}",
     rf"$\lambda_{{\rm large}}$" + f"\n= {eigenvalues[1]:.1f}"],
    color=TEXT_COL, fontsize=10,
)
ax00.set_ylabel(r"$\lambda_i\,/\,(\lambda_i + \lambda)$", color=TEXT_COL, fontsize=11)
ax00.set_title("Diagonal shrinkage matrix entries\n1 = Keep (OLS),  0 = Fully Shrunk",
               color=TEXT_COL, fontsize=10, pad=8)
ax00.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.4)
ax00.grid(axis="y", color=GRID_COL, alpha=0.5)

bars = ax00.bar([0, 1], [1.0, 1.0], color=EI_COLORS, alpha=0.85, width=0.6,
                edgecolor="white", linewidth=0.8)
bar_texts = [
    ax00.text(0, 1.07, "1.000", ha="center", va="bottom",
              color=TEXT_COL, fontsize=12, fontweight="bold"),
    ax00.text(1, 1.07, "1.000", ha="center", va="bottom",
              color=TEXT_COL, fontsize=12, fontweight="bold"),
]

# ── [0,1]: Shrinkage curves with animated cursor ───────────────────────────────
for j in range(2):
    name = r"\lambda_{\rm small}" if j == 0 else r"\lambda_{\rm large}"
    lbl  = rf"${name}={eigenvalues[j]:.1f}$"
    ax01.semilogx(lambdas_full, shrink_full[:, j],
                  color=EI_COLORS[j], lw=2.5, label=lbl)

ax01.axhline(1.0, color="white", lw=1.0, ls=":", alpha=0.7,
             label=r"OLS limit  ($\lambda\to 0$)")
ax01.axhline(0.0, color="white", lw=1.0, ls=":", alpha=0.7,
             label=r"zero  ($\lambda\to\infty$)")
ax01.set_ylim(-0.06, 1.14)
ax01.set_xlabel(r"$\lambda$  (log scale)", color=TEXT_COL, fontsize=10)
ax01.set_ylabel(r"$\lambda_i\,/\,(\lambda_i + \lambda)$", color=TEXT_COL, fontsize=11)
ax01.set_title("Eigenvalue Magnitude Curves",
               color=TEXT_COL, fontsize=10, pad=8)
ax01.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#30363d",
            labelcolor=TEXT_COL, loc="lower left")
ax01.grid(True, color=GRID_COL, alpha=0.5)

cursor_vline = ax01.axvline(lambdas_anim[0], color="white", lw=1.5, ls="--",
                             alpha=0.85, zorder=5)
cursor_dots  = [
    ax01.scatter([lambdas_anim[0]],
                 [eigenvalues[j] / (eigenvalues[j] + lambdas_anim[0])],
                 color=EI_COLORS[j], s=90, zorder=6,
                 edgecolors="white", linewidths=0.8)
    for j in range(2)
]

# ── [1,0]: Solution path in weight space ───────────────────────────────────────
ax10.contourf(W1G, W2G, LOSS,
              levels=np.linspace(LOSS.min(), LOSS.max(), 35),
              cmap="Blues", alpha=0.22, zorder=0)
ax10.contour(W1G, W2G, LOSS, levels=contour_levels,
             colors="#4a90d9", linewidths=0.9, alpha=0.5, zorder=1)
ax10.plot(w_path[:, 0], w_path[:, 1],
          color="#888888", lw=1.5, alpha=0.55, zorder=2)
ax10.scatter(*w_ols, color="#f1c40f", s=150, marker="*", zorder=8,
             edgecolors="white", linewidths=1.2, label=r"$w_{\rm OLS}$")
ax10.scatter(0, 0, color="grey", s=80, marker="x", linewidths=2.5, zorder=6)
ax10.text(0.13, 0.06, r"$\lambda\to\infty$", fontsize=9, color="grey")

path_dot, = ax10.plot([], [], "o", color="#ff6b6b", ms=11,
                      markeredgecolor="white", markeredgewidth=1.2,
                      zorder=9, label=r"$w_{\rm ridge}$")
ax10.set_xlabel(r"$w_1$", color=TEXT_COL, fontsize=12)
ax10.set_ylabel(r"$w_2$", color=TEXT_COL, fontsize=12)
ax10.set_title(r"Solution path $w_{\rm ridge}$ in weight space",
               color=TEXT_COL, fontsize=10, pad=8)
ax10.legend(fontsize=9, facecolor=PANEL_BG, edgecolor="#30363d", labelcolor=TEXT_COL)
ax10.set_aspect("equal")
ax10.grid(True, color=GRID_COL, alpha=0.3)

# ── [1,1]: Regression plot (y vs x₁, x₂ held at mean = 0) ────────────────────
ax11.scatter(x1_vals, y, s=18, color="#4a9edd", alpha=0.45, zorder=2, label="Data")
ax11.plot(x1_plot, w_ols[0] * x1_plot, color="#f1c40f", lw=1.5, ls="--",
          alpha=0.55, zorder=3, label=r"OLS fit")

fit_line, = ax11.plot([], [], color="#ff6b6b", lw=2.5, zorder=4, label="Ridge fit")

y_preds_all = np.array([ridge_w(l)[0] * x1_plot for l in lambdas_anim])
ax11.set_xlim(x1_plot[0], x1_plot[-1])
ax11.set_ylim(min(y.min(), y_preds_all.min()) - 0.4,
              max(y.max(), y_preds_all.max()) + 0.4)
ax11.set_xlabel(r"$x_1$", color=TEXT_COL, fontsize=12)
ax11.set_ylabel(r"$y$", color=TEXT_COL, fontsize=12)
ax11.set_title(r"Fitted Values — Larger $\lambda$ Shrinks the Slope",
               color=TEXT_COL, fontsize=10, pad=8)
ax11.legend(fontsize=9, facecolor=PANEL_BG, edgecolor="#30363d",
            labelcolor=TEXT_COL, loc="upper left")
ax11.grid(True, color=GRID_COL, alpha=0.3)


# ── Update function ────────────────────────────────────────────────────────────
def update(frame):
    lam = lambdas_anim[frame]
    w_r = ridge_w(lam)
    sf  = eigenvalues / (eigenvalues + lam)

    # [0,0] shrinkage bars
    for bar, s in zip(bars, sf):
        bar.set_height(s)
    for txt, s, x in zip(bar_texts, sf, [0, 1]):
        txt.set_position((x, max(s + 0.04, 0.07)))
        txt.set_text(f"{s:.3f}")

    # [0,1] cursor
    cursor_vline.set_xdata([lam, lam])
    for j, dot in enumerate(cursor_dots):
        dot.set_offsets([[lam, sf[j]]])

    # [1,0] path dot
    path_dot.set_data([w_r[0]], [w_r[1]])

    # [1,1] regression line (slope = w_r[0], x2 at mean = 0)
    fit_line.set_data(x1_plot, w_r[0] * x1_plot)

    # title λ readout
    super_title.set_text(rf"$\lambda={lam:.2f}$")

    return [*bars, *bar_texts, cursor_vline, *cursor_dots,
            path_dot, fit_line, super_title]


anim = animation.FuncAnimation(
    fig, update, frames=N_FRAMES, interval=80, blit=False,
)

stamp    = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
out_path = OUTPUT_DIR / f"{stamp}_ridge_lambda_animation.gif"
anim.save(str(out_path), writer="pillow", fps=12, dpi=100)
print(f"Saved to: {out_path}")
plt.show()
