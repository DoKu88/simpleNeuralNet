"""
lasso_ridge_dataset_gifs.py

Two boomerang GIFs — one for Lasso, one for Ridge — showing how the
dataset view from lasso_ridge_dataset_overview.py changes with λ.

Left panel:  bar chart of regularised weights w(λ)
             (ghost bars show OLS reference at λ=0)
Right panel: fitted vs observed scatter  y_hat(λ) = X @ w(λ)  vs  y
             (axis limits fixed to OLS extent so drift is visible)

Lasso:  w_i(λ) = max(0, w_OLS_i − λ/2)   soft-thresholding, sparsity
Ridge:  w_i(λ) = w_OLS_i / (1 + λ)        proportional shrinkage
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET  (identical to lasso_ridge_dataset_overview.py)
# ══════════════════════════════════════════════════════════════════════════════
W_OLS      = np.array([3.0, 2.2, 1.8, 1.2, 0.9, 0.6, 0.4, 0.25, 0.15, 0.08])
n_features = len(W_OLS)

np.random.seed(42)
n_samples = 50
sigma     = 0.5

X_raw = np.random.randn(n_samples, n_features)
Q, _  = np.linalg.qr(X_raw)
X     = Q

y         = X @ W_OLS + sigma * np.random.randn(n_samples)
y_hat_ols = X @ W_OLS

# ══════════════════════════════════════════════════════════════════════════════
#  λ SEQUENCE  (quadratic boomerang: dense near 0 where action happens)
# ══════════════════════════════════════════════════════════════════════════════
LAM_MAX = 7.0
N_HALF  = 55
PAUSE   = 12

_t       = np.linspace(0, 1, N_HALF)
lam_up   = LAM_MAX * _t ** 2
lam_down = LAM_MAX * _t[::-1] ** 2
LAMBDAS  = np.concatenate([
    np.full(PAUSE, 0.0), lam_up[1:],
    np.full(PAUSE, LAM_MAX), lam_down[1:],
])

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════════════════
COLORS    = [plt.cm.tab10(i) for i in range(n_features)]
GRID_CLR  = "#ebebeb"
LASSO_CLR = "#e74c3c"
RIDGE_CLR = "#2563eb"

# Fixed scatter axis limits (based on OLS, so drift toward 0 is visible)
PAD   = 0.5
lo_x  = y_hat_ols.min() - PAD
hi_x  = y_hat_ols.max() + PAD
lo_y  = y.min() - PAD
hi_y  = y.max() + PAD

def _style(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.grid(True, color=GRID_CLR, zorder=0)
    ax.tick_params(labelsize=10)


# ══════════════════════════════════════════════════════════════════════════════
#  GIF FACTORY
# ══════════════════════════════════════════════════════════════════════════════
def make_gif(method):
    is_lasso  = method == "lasso"
    clr       = LASSO_CLR if is_lasso else RIDGE_CLR
    label     = "L1 Lasso" if is_lasso else "L2 Ridge"
    rule      = r"$w_i(\lambda)=\max(0,\,w_i^{\mathrm{OLS}}-\lambda/2)$" if is_lasso \
                else r"$w_i(\lambda)=w_i^{\mathrm{OLS}}\,/\,(1+\lambda)$"

    fig, (ax_w, ax_y) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.13, wspace=0.30)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    _style(ax_w)
    # Ghost OLS reference bars
    ax_w.bar(np.arange(1, n_features + 1), W_OLS,
             color=COLORS, alpha=0.15, zorder=2, edgecolor="none")
    # Animated bars
    bars = ax_w.bar(np.arange(1, n_features + 1), W_OLS,
                    color=COLORS, alpha=0.85, zorder=3,
                    edgecolor="white", linewidth=0.5)
    bar_labels = [
        ax_w.text(i + 1, W_OLS[i] + 0.06, str(W_OLS[i]),
                  ha="center", va="bottom", fontsize=7.5, color="#444", zorder=4)
        for i in range(n_features)
    ]
    ax_w.set_xlabel("Feature Index", fontsize=12)
    ax_w.set_ylabel("Weight Value", fontsize=12)
    ax_w.set_title(r"Regularised Weights $w(\lambda)$"
                   "\n(ghost bars = OLS at λ = 0)",
                   fontsize=11, fontweight="bold", pad=6)
    ax_w.set_xticks(np.arange(1, n_features + 1))
    ax_w.set_ylim(0, W_OLS[0] * 1.25)

    # ── Scatter ───────────────────────────────────────────────────────────────
    _style(ax_y)
    scat = ax_y.scatter(y_hat_ols, y, color=clr, alpha=0.75, s=48, zorder=3,
                        edgecolors="white", linewidths=0.5)
    ax_y.plot([lo_x, hi_x], [lo_x, hi_x],
              color="#888", lw=1.4, ls="--", zorder=2, label="Perfect fit")
    ax_y.set_xlim(lo_x, hi_x)
    ax_y.set_ylim(lo_y, hi_y)
    ax_y.set_xlabel(r"Fitted  $\hat{y}(\lambda) = Xw(\lambda)$", fontsize=12)
    ax_y.set_ylabel(r"Observed  $y$", fontsize=12)
    ax_y.set_title("Fitted vs Observed\n(axis fixed to OLS extent)",
                   fontsize=11, fontweight="bold", pad=6)
    ax_y.legend(fontsize=10, framealpha=0.9)

    # Figure-level titles (updated each frame)
    sup  = fig.suptitle("", fontsize=13, fontweight="bold", y=0.99)
    sub  = fig.text(0.5, 0.92, "", ha="center", va="top", fontsize=10, color="#444")

    def animate(frame_idx):
        lam = LAMBDAS[frame_idx]
        w   = (np.maximum(0.0, W_OLS - lam / 2) if is_lasso
               else W_OLS / (1.0 + lam))
        yh  = X @ w

        # Update bars + labels
        for bar, lbl, h, h_ols in zip(bars, bar_labels, w, W_OLS):
            bar.set_height(h)
            lbl.set_position((lbl.get_position()[0], h + 0.06))
            lbl.set_text(f"{h:.2f}" if h > 0.005 else "")

        # Update scatter
        scat.set_offsets(np.c_[yh, y])

        # Titles
        sup.set_text(rf"{label}   $\lambda = {lam:.2f}$")
        sub.set_text(
            rule +
            rf"     $w \in \mathbb{{R}}^{{{n_features}}}$,"
            rf"  $X \in \mathbb{{R}}^{{{n_samples}\times{n_features}}}$,"
            rf"  $y \in \mathbb{{R}}^{{{n_samples}}}$"
        )

        if frame_idx % 15 == 0:
            print(f"  [{label}] frame {frame_idx+1}/{len(LAMBDAS)}  λ={lam:.2f}")

    ani = animation.FuncAnimation(fig, animate, frames=len(LAMBDAS),
                                  interval=60, blit=False)
    out_path = OUTPUT_DIR / f"{stamp}_{method}_dataset_boomerang.gif"
    print(f"\nRendering {label} GIF…")
    ani.save(str(out_path), writer="pillow", fps=15, dpi=100)
    print(f"Saved: {out_path}")
    plt.close(fig)


make_gif("lasso")
make_gif("ridge")
