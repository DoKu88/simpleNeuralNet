"""
lasso_ridge_compare_boomerang.py

Side-by-side L1 Lasso (left) vs L2 Ridge (right) constraint diagrams.
Shared λ sweeps low → high → low (boomerang GIF).
Bottom bar shows the shared λ value.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle, Polygon, Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY  (shared between both panels)
# ══════════════════════════════════════════════════════════════════════════════
BETA_OLS       = np.array([0.5, 2.3])
TILT_DEG       = 27.0
ASPECT         = 3.65
ELLIPSE_LEVELS = [0.35, 0.85, 1.8, 3.5]

tilt_rad = np.radians(TILT_DEG)
ct, st   = np.cos(tilt_rad), np.sin(tilt_rad)
R_rot    = np.array([[ct, -st], [st, ct]])
A_mat    = R_rot @ np.diag([1.0 / ASPECT**2, 1.0]) @ R_rot.T
eig_vals, eig_vecs = np.linalg.eigh(A_mat)
patch_angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))

def ellipse_wh(c):
    return 2.0 * np.sqrt(c / eig_vals[0]), 2.0 * np.sqrt(c / eig_vals[1])

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRAINT SCALES
# ══════════════════════════════════════════════════════════════════════════════
L2_OLS = float(np.linalg.norm(BETA_OLS))     # ≈ 2.524
L1_OLS = float(np.sum(np.abs(BETA_OLS)))      # = 3.500

# At λ=0 both constraints exactly touch OLS (non-binding), shrink to 0 as λ→∞.
# r(λ) = L2_OLS/(1+λ),  t(λ) = L1_OLS/(1+λ)  — constraint never exceeds OLS norm.
LAM_MIN, LAM_MAX = 0.0, 50.0

def ridge_r(lam): return L2_OLS / (1.0 + lam)
def lasso_t(lam): return L1_OLS / (1.0 + lam)

# ══════════════════════════════════════════════════════════════════════════════
#  ESTIMATE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def ridge_estimate(r):
    """Nearest Euclidean point on the L2 circle to OLS — radial projection."""
    if L2_OLS <= r:
        return BETA_OLS.copy()
    return r * BETA_OLS / L2_OLS

def lasso_estimate(t):
    """Nearest Euclidean point on the L1 diamond to OLS (OLS assumed in Q1)."""
    if L1_OLS <= t:
        return BETA_OLS.copy()
    # Project OLS onto the Q1 edge  w1+w2=t  via uniform subtraction
    d = (L1_OLS - t) / 2.0
    w = BETA_OLS - d
    # If either coord goes negative, snap to the nearest axis corner
    if w[0] < 0:
        return np.array([0.0, t])
    if w[1] < 0:
        return np.array([t, 0.0])
    return w

def diamond_verts(t):
    return np.array([[t, 0], [0, t], [-t, 0], [0, -t]])

# ══════════════════════════════════════════════════════════════════════════════
#  LAMBDA SEQUENCE  (boomerang: low → high → low)
# ══════════════════════════════════════════════════════════════════════════════
# Quadratic spacing: dense frames near λ=0 where the geometry changes fastest,
# sparse near λ=LAM_MAX where both constraints are already near-zero.
N_HALF, PAUSE = 65, 12
_t        = np.linspace(0, 1, N_HALF)
lam_up    = LAM_MAX * _t**2          # 0 → LAM_MAX, quadratic
lam_down  = LAM_MAX * _t[::-1]**2   # LAM_MAX → 0, quadratic
LAMBDAS = np.concatenate([
    np.full(PAUSE, 0.0),
    lam_up[1:],           # exclude duplicate 0
    np.full(PAUSE, LAM_MAX),
    lam_down[1:],         # exclude duplicate LAM_MAX
])
N_FRAMES = len(LAMBDAS)

# ── Precompute per-frame data ──────────────────────────────────────────────────
print(f"Precomputing {N_FRAMES} frames…")
frame_data = [
    (lam, ridge_r(lam), lasso_t(lam),
     ridge_estimate(ridge_r(lam)),
     lasso_estimate(lasso_t(lam)))
    for lam in LAMBDAS
]
print("Done.")

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════════════════
ELLIPSE_FILL    = "#c8e6c9"
ELLIPSE_EDGE    = "#2e7d32"
ELLIPSE_ALPHAS  = [0.70, 0.58, 0.46, 0.34]
CONSTRAINT_CLR  = "#4472C4"
CONSTRAINT_A    = 0.55
DIAMOND_EDGE    = "#cc0000"
DASH_CLR        = "#E65C00"
AX_CLR          = "black"
LBL_FS, AX_FS, TITLE_FS = 10, 13, 11

XLIM = (-3.0, 3.5)
YLIM = (-1.5, 4.0)

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 7.5))
fig.patch.set_facecolor("white")
gs  = GridSpec(2, 2, figure=fig,
               height_ratios=[10, 1],
               hspace=0.10, wspace=0.08,
               left=0.04, right=0.97, top=0.91, bottom=0.10)
ax_l  = fig.add_subplot(gs[0, 0])
ax_r  = fig.add_subplot(gs[0, 1])
ax_sl = fig.add_subplot(gs[1, :])

lam_title = fig.text(0.5, 0.97, rf"$\lambda = {LAM_MIN:.2f}$",
                     ha="center", va="top", fontsize=15, fontweight="bold")


def _init_ax(ax, title, eq):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_title(eq + "\n" + title, fontsize=TITLE_FS, pad=4)
    _a = dict(arrowstyle="-|>", color=AX_CLR, lw=1.5, mutation_scale=12)
    ax.annotate("", xy=(XLIM[1]-0.12, 0), xytext=(XLIM[0]+0.08, 0),
                arrowprops=_a, zorder=7)
    ax.annotate("", xy=(0, YLIM[1]-0.08), xytext=(0, YLIM[0]+0.08),
                arrowprops=_a, zorder=7)
    ax.text(XLIM[1]-0.08, -0.18, r"$w_1$", fontsize=AX_FS,
            ha="left", va="top", style="italic")
    ax.text(-0.14, YLIM[1]-0.08, r"$w_2$", fontsize=AX_FS,
            ha="right", va="center", style="italic")

_init_ax(ax_l, "L1  Lasso",  r"$L(w) = \|y-Xw\|^2 + \lambda\|w\|_1$")
_init_ax(ax_r, "L2  Ridge",  r"$L(w) = \|y-Xw\|^2 + \lambda\|w\|^2$")

# ── Static RSS ellipses + OLS dot (both panels) ───────────────────────────────
for ax in (ax_l, ax_r):
    for lvl, alpha in zip(reversed(ELLIPSE_LEVELS), reversed(ELLIPSE_ALPHAS)):
        w, h = ellipse_wh(lvl)
        ax.add_patch(Ellipse(
            xy=BETA_OLS, width=w, height=h, angle=patch_angle,
            facecolor=ELLIPSE_FILL, alpha=alpha,
            edgecolor=ELLIPSE_EDGE, linewidth=1.2, zorder=5, clip_on=True,
        ))
    ax.scatter(*BETA_OLS, color="black", s=60, zorder=11)
    ax.annotate(
        "OLS estimate",
        xy=BETA_OLS,
        xytext=BETA_OLS + np.array([0.25, 0.50]),
        fontsize=LBL_FS, color="black", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.9, mutation_scale=10),
        ha="left", va="bottom", zorder=12,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  ANIMATED ARTISTS
# ══════════════════════════════════════════════════════════════════════════════
lam0, r0, t0, br0, bl0 = frame_data[0]

# ── Lasso: diamond ────────────────────────────────────────────────────────────
diamond_patch = Polygon(diamond_verts(t0), closed=True,
                        facecolor=CONSTRAINT_CLR, alpha=CONSTRAINT_A,
                        edgecolor=DIAMOND_EDGE, linewidth=2.0, zorder=6, clip_on=True)
ax_l.add_patch(diamond_patch)

(l_dash,)    = ax_l.plot([], [], color=DASH_CLR, lw=1.8,
                          linestyle=(0, (6, 4)), zorder=6, alpha=0.0)
(l_dot,)     = ax_l.plot([], [], "o", color="black", ms=8, zorder=11, alpha=0.0)
l_lbl        = ax_l.text(0, 0, "Lasso estimate", fontsize=LBL_FS,
                          color="black", fontweight="bold",
                          ha="right", va="bottom", zorder=12, alpha=0.0)
(l_lbl_con,) = ax_l.plot([], [], color="black", lw=0.8, zorder=12, alpha=0.0)

# ── Ridge: disk ───────────────────────────────────────────────────────────────
disk_patch   = Circle((0, 0), r0, facecolor=CONSTRAINT_CLR, alpha=CONSTRAINT_A,
                       edgecolor=CONSTRAINT_CLR, linewidth=1.5, zorder=6, clip_on=True)
ax_r.add_patch(disk_patch)

(r_dash,)    = ax_r.plot([], [], color=DASH_CLR, lw=1.8,
                          linestyle=(0, (6, 4)), zorder=6, alpha=0.0)
(r_dot,)     = ax_r.plot([], [], "o", color="black", ms=8, zorder=11, alpha=0.0)
r_lbl        = ax_r.text(0, 0, "Ridge estimate", fontsize=LBL_FS,
                          color="black", fontweight="bold",
                          ha="right", va="bottom", zorder=12, alpha=0.0)
(r_lbl_con,) = ax_r.plot([], [], color="black", lw=0.8, zorder=12, alpha=0.0)

# ── Lambda bar ────────────────────────────────────────────────────────────────
ax_sl.set_facecolor("#f5f5f5")
for sp in ax_sl.spines.values():
    sp.set_edgecolor("#cccccc")
ax_sl.set_xlim(LAM_MIN, LAM_MAX)
ax_sl.set_ylim(0, 1)
ax_sl.set_xticks([])
ax_sl.set_yticks([])

bar_rect    = Rectangle((LAM_MIN, 0), 0.0, 1,
                         color=CONSTRAINT_CLR, alpha=0.7, zorder=2)
ax_sl.add_patch(bar_rect)
(bar_line,) = ax_sl.plot([LAM_MIN, LAM_MIN], [0, 1], color="black", lw=2, alpha=0.9)
bar_txt     = ax_sl.text(0.995, 0.5, f"{LAM_MIN:.2f}",
                          ha="right", va="center", fontsize=11,
                          transform=ax_sl.transAxes)
ax_sl.text(-0.01, 0.5, r"$\lambda$", ha="right", va="center",
           fontsize=13, transform=ax_sl.transAxes, fontweight="bold")
ax_sl.text(0.0, -0.22, "small λ  (less shrinkage)",
           ha="left", va="top", fontsize=8, color="#555555",
           transform=ax_sl.transAxes)
ax_sl.text(1.0, -0.22, "large λ  (more shrinkage)",
           ha="right", va="top", fontsize=8, color="#555555",
           transform=ax_sl.transAxes)

# ══════════════════════════════════════════════════════════════════════════════
#  UPDATE
# ══════════════════════════════════════════════════════════════════════════════
_LBL_OFF = np.array([-0.85, 0.70])

def update(frame):
    lam, r, t, br, bl = frame_data[frame]
    l_bind = L1_OLS > t
    r_bind = L2_OLS > r

    # ── Lasso panel ───────────────────────────────────────────────────────────
    diamond_patch.set_xy(diamond_verts(t))
    diamond_patch.set_alpha(CONSTRAINT_A)
    diamond_patch.set_edgecolor(DIAMOND_EDGE)

    l_dot.set_data([bl[0]], [bl[1]])
    l_dot.set_alpha(1.0)
    lp = bl + _LBL_OFF
    l_lbl.set_position(lp)
    l_lbl.set_alpha(1.0)
    l_lbl_con.set_data([lp[0], bl[0]], [lp[1], bl[1]])
    l_lbl_con.set_alpha(0.7)
    if l_bind:
        l_dash.set_data([BETA_OLS[0], bl[0]], [BETA_OLS[1], bl[1]])
        l_dash.set_alpha(0.9)
    else:
        l_dash.set_alpha(0.0)

    # ── Ridge panel ───────────────────────────────────────────────────────────
    disk_patch.set_radius(r)
    disk_patch.set_alpha(CONSTRAINT_A)

    r_dot.set_data([br[0]], [br[1]])
    r_dot.set_alpha(1.0)
    rp = br + _LBL_OFF
    r_lbl.set_position(rp)
    r_lbl.set_alpha(1.0)
    r_lbl_con.set_data([rp[0], br[0]], [rp[1], br[1]])
    r_lbl_con.set_alpha(0.7)
    if r_bind:
        r_dash.set_data([BETA_OLS[0], br[0]], [BETA_OLS[1], br[1]])
        r_dash.set_alpha(0.9)
    else:
        r_dash.set_alpha(0.0)

    # ── Lambda bar ────────────────────────────────────────────────────────────
    bar_rect.set_width(lam - LAM_MIN)
    bar_line.set_data([lam, lam], [0, 1])
    bar_txt.set_text(f"{lam:.2f}")
    lam_title.set_text(rf"$\lambda = {lam:.2f}$")

    if frame % 25 == 0:
        print(f"  frame {frame+1}/{N_FRAMES}  λ={lam:.2f}")

    return [diamond_patch, l_dash, l_dot, l_lbl, l_lbl_con,
            disk_patch, r_dash, r_dot, r_lbl, r_lbl_con,
            bar_rect, bar_line, bar_txt, lam_title]


ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=70, blit=False)

stamp    = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
out_path = OUTPUT_DIR / f"{stamp}_lasso_ridge_compare.gif"
print(f"Saving → {out_path}")
ani.save(str(out_path), writer="pillow", fps=15, dpi=110)
print(f"Saved: {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  WEIGHT PATH PLOT  (static PNG)
# ══════════════════════════════════════════════════════════════════════════════
PLOT_LAM_MAX = 12.0
lam_dense    = np.linspace(0, PLOT_LAM_MAX, 800)

# Ridge: w_i(λ) = OLS_i / (1+λ)  (closed form from radial projection)
ridge_w1_path = BETA_OLS[0] / (1.0 + lam_dense)
ridge_w2_path = BETA_OLS[1] / (1.0 + lam_dense)

# Lasso: via existing estimate function
lasso_ws      = np.array([lasso_estimate(lasso_t(lam)) for lam in lam_dense])
lasso_w1_path = lasso_ws[:, 0]
lasso_w2_path = lasso_ws[:, 1]

# λ at which Lasso w1 first hits 0  (analytical: solve d = BETA_OLS[0])
lasso_zero_lam = 2.0 * BETA_OLS[0] / (L1_OLS - 2.0 * BETA_OLS[0])

W1_CLR   = "#e74c3c"
W2_CLR   = "#2563eb"
GRID_CLR = "#ebebeb"

fig2, (ax_lp, ax_rp) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig2.patch.set_facecolor("white")
fig2.suptitle(
    r"Regularization Path — weight magnitudes vs $\lambda$",
    fontsize=14, fontweight="bold", y=0.99,
)
plt.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.14, wspace=0.06)

for ax, w1, w2, title in [
    (ax_lp, lasso_w1_path, lasso_w2_path, "L1  Lasso"),
    (ax_rp, ridge_w1_path, ridge_w2_path, "L2  Ridge"),
]:
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.grid(True, color=GRID_CLR, zorder=0)
    ax.plot(lam_dense, w1, color=W1_CLR, lw=2.5, label=r"$w_1$", zorder=3)
    ax.plot(lam_dense, w2, color=W2_CLR, lw=2.5, label=r"$w_2$", zorder=3)
    ax.axhline(0, color="#aaaaaa", lw=0.8, zorder=1)
    ax.set_xlabel(r"$\lambda$", fontsize=13)
    ax.set_xlim(0, PLOT_LAM_MAX)
    ax.set_ylim(-0.08, BETA_OLS[1] * 1.1)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.legend(fontsize=12, framealpha=0.9, loc="upper right")
    ax.tick_params(labelsize=10)

ax_lp.set_ylabel("weight value", fontsize=12)

# Mark the λ where Lasso w1 reaches exactly 0
ax_lp.axvline(lasso_zero_lam, color=W1_CLR, lw=1.3, ls="--", alpha=0.65, zorder=2)
ax_lp.text(
    lasso_zero_lam + 0.18, BETA_OLS[1] * 1.02,
    f"w₁ = 0,  λ ≈ {lasso_zero_lam:.2f}",
    color=W1_CLR, fontsize=9, va="top", ha="left", fontweight="bold",
)

out_path2 = OUTPUT_DIR / f"{stamp}_lasso_ridge_weight_paths.png"
fig2.savefig(str(out_path2), dpi=140, bbox_inches="tight", facecolor="white")
print(f"Saved weight path plot: {out_path2}")

plt.show()
