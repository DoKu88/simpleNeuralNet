"""
linear_regression_MLE_normal.py

Single-panel figure showing that ordinary least-squares linear regression IS
the maximum-likelihood estimator (MLE) when noise is Gaussian.

  - 60 points sampled from  y = x + ε,  ε ~ N(0, σ²)
  - OLS fitted line (= MLE solution)
  - Continuous gradient band along the line encoding the Gaussian probability
    density: opaque near the line (high probability) fading to transparent at
    ±3σ (low probability)
  - Explicit Gaussian pdf curves at three x-values to make the distribution
    shape unambiguous
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import norm

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── data generation ───────────────────────────────────────────────────────────
rng            = np.random.default_rng(42)
TRUE_SLOPE     = 1.0
TRUE_INTERCEPT = 0.0
SIGMA_TRUE     = 1.5
N              = 60

x = rng.uniform(0, 10, N)
y = TRUE_SLOPE * x + TRUE_INTERCEPT + rng.normal(0, SIGMA_TRUE, N)

# ── OLS fit (identical to MLE under Gaussian noise) ───────────────────────────
X_mat    = np.column_stack([np.ones(N), x])
beta, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
b0_mle, b1_mle = beta

residuals = y - (b0_mle + b1_mle * x)
sigma_mle = np.sqrt(np.mean(residuals ** 2))

# ── aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.size": 13,
})

C_DATA  = "#0072B2"   # blue
C_LINE  = "#D55E00"   # orange-red
C_TRUE  = "#0072B2"   # blue
C_GAUSS = "#2CA02C"   # green
C_SIGMA = "#9467BD"   # purple for sigma labels

fig, ax = plt.subplots(figsize=(13, 8), facecolor="white")

fig.suptitle(
    "Linear Regression as MLE under Gaussian Noise",
    fontsize=16, fontweight="bold", y=0.98,
)

# ── line values across the plot range ─────────────────────────────────────────
x_plt = np.linspace(-0.3, 10.3, 400)
y_fit = b1_mle * x_plt + b0_mle

# ── continuous gradient band: Gaussian density encoded as opacity ─────────────
# Divide [0, 3σ] into many thin slices; each slice's opacity mirrors the pdf.
N_BANDS   = 80
max_sigma = 3.2
y_vals    = np.linspace(-max_sigma * sigma_mle, max_sigma * sigma_mle, 500)
pdf_vals  = norm.pdf(y_vals, 0, sigma_mle)
pdf_max   = pdf_vals.max()

for i in range(N_BANDS, 0, -1):
    lo = (i - 1) / N_BANDS * max_sigma * sigma_mle
    hi = i       / N_BANDS * max_sigma * sigma_mle
    # average pdf over this thin slice → drives opacity
    mid    = (lo + hi) / 2.0
    alpha  = norm.pdf(mid, 0, sigma_mle) / pdf_max * 0.55
    ax.fill_between(x_plt,
                    y_fit - hi, y_fit - lo,
                    alpha=alpha, color=C_GAUSS, linewidth=0)
    ax.fill_between(x_plt,
                    y_fit + lo, y_fit + hi,
                    alpha=alpha, color=C_GAUSS, linewidth=0)

# ── 1σ and 2σ dashed envelope lines ──────────────────────────────────────────
for k, ls, lw in [(1, "--", 1.2), (2, ":", 1.0)]:
    ax.plot(x_plt, y_fit + k * sigma_mle, color=C_GAUSS,
            ls=ls, lw=lw, alpha=0.55)
    ax.plot(x_plt, y_fit - k * sigma_mle, color=C_GAUSS,
            ls=ls, lw=lw, alpha=0.55)

# Label 0σ on the fit line, +kσ above, -kσ below
ax.text(10.35, y_fit[-1], r"$0\sigma$",
        color=C_SIGMA, fontsize=11, va="center", fontweight="bold")
for k in [1, 2]:
    ax.text(10.35, y_fit[-1] + k * sigma_mle, rf"$+{k}\sigma$",
            color=C_SIGMA, fontsize=11, va="center", fontweight="bold")
    ax.text(10.35, y_fit[-1] - k * sigma_mle, rf"$-{k}\sigma$",
            color=C_SIGMA, fontsize=11, va="center", fontweight="bold")

# ── true line, fitted line, data ──────────────────────────────────────────────
ax.plot(x_plt, TRUE_SLOPE * x_plt + TRUE_INTERCEPT,
        color=C_TRUE, lw=2.0, ls="-", alpha=0.85, label="True line:  y = x")
ax.plot(x_plt, y_fit, color=C_LINE, lw=2.5, zorder=4,
        label=rf"MLE / OLS fit:  $\hat{{y}} = {b1_mle:.2f}x {b0_mle:+.2f}$")
ax.scatter(x, y, color=C_DATA, s=38, alpha=0.80, zorder=5, label="Observed data")

ax.set_xlim(-0.5, 11.8)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
ax.set_title(
    r"$y = \beta_1 x + \beta_0 + \varepsilon$,   "
    r"$\varepsilon \sim \mathrm{N}(0,\, \sigma^2)$"
    rf"     (fitted $\hat{{\sigma}} = {sigma_mle:.2f}$)",
    fontsize=12, pad=8,
)
gauss_patch = Patch(facecolor=C_GAUSS, alpha=0.45,
                    label=r"Normal dist. $\mathrm{N}(\hat{y},\,\sigma^2)$ — darker = more probable")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=[gauss_patch] + handles, labels=[gauss_patch.get_label()] + labels,
          fontsize=11, loc="lower right", framealpha=0.9)

plt.tight_layout()
out_path = OUTPUT_DIR / "linear_regression_MLE_normal.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()

# ═════════════════════════════════════════════════════════════════════════════
# THUMBNAIL — Option 2: Residuals view  (horizontal line at y=0)
# ═════════════════════════════════════════════════════════════════════════════

def _gradient_band(ax, x_arr, y_center, sigma, n_bands=80, max_s=3.2, color=C_GAUSS):
    """Fill a Gaussian-density gradient band around y_center."""
    pdf_peak = norm.pdf(0, 0, sigma)
    for i in range(n_bands, 0, -1):
        lo  = (i - 1) / n_bands * max_s * sigma
        hi  = i       / n_bands * max_s * sigma
        mid = (lo + hi) / 2.0
        a   = norm.pdf(mid, 0, sigma) / pdf_peak * 0.80
        ax.fill_between(x_arr, y_center - hi, y_center - lo,
                        alpha=a, color=color, linewidth=0)
        ax.fill_between(x_arr, y_center + lo, y_center + hi,
                        alpha=a, color=color, linewidth=0)

def _vertical_bell(ax, x_anchor, y_center, sigma, width=1.6, color=C_GAUSS,
                   outline="white", fit_color=C_LINE):
    """Draw a vertical Gaussian bell to the right of x_anchor, centered at y_center."""
    y_b   = np.linspace(y_center - 3.2 * sigma, y_center + 3.2 * sigma, 300)
    pdf_b = norm.pdf(y_b, y_center, sigma)
    sc    = pdf_b / pdf_b.max() * width
    ax.fill_betweenx(y_b, x_anchor, x_anchor + sc,
                     alpha=0.55, color=color, linewidth=0, zorder=3)
    ax.plot(x_anchor + sc, y_b, color=outline, lw=1.6, alpha=0.90, zorder=4)
    ax.plot([x_anchor - 0.15, x_anchor + 0.15], [y_center, y_center],
            color=fit_color, lw=2.0, zorder=6)

x_plot = np.linspace(0, 10, 300)

# ── residuals thumbnail ───────────────────────────────────────────────────────
RBG      = "white"
RC_TEXT  = "#111111"
RC_TICK  = "#444444"
RC_GAUSS = "#2CA02C"
RC_FIT   = "#D55E00"
RC_DATA  = "#0072B2"

fig_r, ax_r = plt.subplots(figsize=(8, 4.2), facecolor=RBG)
ax_r.set_facecolor(RBG)

_gradient_band(ax_r, x_plot, 0, sigma_mle, color=RC_GAUSS)

# gradient bell on the right — same density-to-opacity mapping as the band
_bell_x    = 10.6
_bell_width = 1.8
_pdf_peak  = norm.pdf(0, 0, sigma_mle)
_y_slices  = np.linspace(-3.2 * sigma_mle, 3.2 * sigma_mle, 300)
for _j in range(len(_y_slices) - 1):
    _ymid  = (_y_slices[_j] + _y_slices[_j + 1]) / 2.0
    _a     = norm.pdf(_ymid, 0, sigma_mle) / _pdf_peak * 0.80
    _x_hi  = _bell_x + norm.pdf(_ymid, 0, sigma_mle) / _pdf_peak * _bell_width
    ax_r.fill_between([_bell_x, _x_hi],
                      [_y_slices[_j]] * 2, [_y_slices[_j + 1]] * 2,
                      alpha=_a, color=RC_GAUSS, linewidth=0, zorder=3)
# outline only
_y_out  = np.linspace(-3.2 * sigma_mle, 3.2 * sigma_mle, 300)
_x_out  = _bell_x + norm.pdf(_y_out, 0, sigma_mle) / _pdf_peak * _bell_width
ax_r.plot(_x_out, _y_out, color="#1a6e1a", lw=1.4, alpha=0.9, zorder=4)

ax_r.axhline(0, color=RC_FIT, lw=2.5, zorder=4)
ax_r.scatter(x, residuals, color=RC_DATA, s=22, alpha=0.75, zorder=5)

for xi, ri in zip(x, residuals):
    ax_r.plot([xi, xi], [0, ri], color=RC_DATA, lw=0.6, alpha=0.35, zorder=2)

ax_r.annotate(r"$\mathrm{N}(0,\,\sigma^2)$",
              xy=(12.3, sigma_mle * 1.4),
              fontsize=9, color=RC_TEXT, fontweight="bold",
              ha="right", va="bottom", zorder=7)

ax_r.set_xlim(-0.5, 13.5)
ax_r.set_ylim(-3.8 * sigma_mle, 3.8 * sigma_mle)
ax_r.set_xlabel("x", fontsize=11, color=RC_TICK)
ax_r.set_ylabel(r"Residual  $\varepsilon = y - \hat{y}$", fontsize=11, color=RC_TICK)
ax_r.tick_params(colors=RC_TICK, labelsize=9)
for spine in ax_r.spines.values():
    spine.set_edgecolor("#cccccc")
ax_r.axhline(0, color=RC_FIT, lw=2.5, zorder=4)   # redraw on top of spines

fig_r.suptitle("Linear Regression  =  Max Likelihood Estimation with Gaussian Noise",
               fontsize=13, fontweight="bold", color=RC_TEXT, y=0.97)
fig_r.text(0.42, 0.02,
           r"$\hat{y} = \beta_1 x + \beta_0 + \varepsilon,$   "
           r"$\varepsilon \sim \mathrm{N}(0,\,\sigma^2)$   "
           r"$\Rightarrow$   minimising $\sum\varepsilon_i^2$ maximises the likelihood",
           ha="center", va="bottom", fontsize=9, color=RC_TEXT)

plt.tight_layout(rect=[0, 0.06, 1, 1.0])
resid_path = OUTPUT_DIR / "linear_regression_MLE_residuals_thumb.png"
plt.savefig(resid_path, dpi=150, bbox_inches="tight", facecolor=RBG)
print(f"Saved → {resid_path}")
plt.show()

# ═════════════════════════════════════════════════════════════════════════════
# THUMBNAIL — Option 1: Horizontal line (estimating a mean)
# ═════════════════════════════════════════════════════════════════════════════

BG       = "#0D1117"
C_T_FIT  = "#FF7B3B"
C_T_DATA = "#79C0FF"
C_T_TEXT = "#E6EDF3"

rng2       = np.random.default_rng(7)
MU_TRUE    = 5.0
SIGMA_H    = 1.6
x_h        = rng2.uniform(0, 10, 60)
y_h        = MU_TRUE + rng2.normal(0, SIGMA_H, 60)
mu_hat     = float(np.mean(y_h))
sigma_h    = float(np.std(y_h, ddof=1))

fig_h, ax_h = plt.subplots(figsize=(8, 4.2), facecolor=BG)
ax_h.set_facecolor(BG)
ax_h.axis("off")

_gradient_band(ax_h, x_plot, mu_hat, sigma_h)
_vertical_bell(ax_h, 10.6, mu_hat, sigma_h)

ax_h.axhline(mu_hat, color=C_T_FIT, lw=3.0, zorder=4)
ax_h.scatter(x_h, y_h, color=C_T_DATA, s=22, alpha=0.65, zorder=5)

# vertical drop lines
for xi, yi in zip(x_h, y_h):
    ax_h.plot([xi, xi], [mu_hat, yi], color=C_T_DATA, lw=0.6, alpha=0.35, zorder=2)

ax_h.annotate(r"$\mathrm{N}(\mu,\,\sigma^2)$",
              xy=(12.45, mu_hat + sigma_h * 0.6),
              fontsize=9, color="white", fontweight="bold",
              ha="center", va="bottom", zorder=7)

ax_h.set_xlim(-0.5, 13.5)
ax_h.set_ylim(mu_hat - 3.8 * sigma_h, mu_hat + 3.8 * sigma_h)

fig_h.text(0.5, 0.93, "Gaussian Data  →  Mean = Maximum Likelihood Estimate",
           ha="center", va="top", fontsize=15, fontweight="bold", color=C_T_TEXT)
fig_h.text(0.5, 0.10,
           r"$y_i \sim \mathrm{N}(\mu,\,\sigma^2)$  "
           r"→  $\hat{\mu} = \bar{y}$ maximises the likelihood",
           ha="center", va="bottom", fontsize=9, color=C_T_TEXT)

plt.tight_layout(rect=[0, 0.08, 1, 0.90])
horiz_path = OUTPUT_DIR / "linear_regression_MLE_horizontal_thumb.png"
plt.savefig(horiz_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {horiz_path}")
plt.show()
