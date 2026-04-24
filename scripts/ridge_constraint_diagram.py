"""
ridge_constraint_diagram.py  (animated GIF)

Ridge Regression constraint diagram animated over λ.
  • OLS RSS contours — fixed nested green ellipses
  • L2 constraint disk — radius r = R_SCALE / √λ, shrinks/grows with λ
  • Ridge estimate — tangency point between disk boundary and RSS ellipses
  • Title: L(w) = ||y − Xw||² + λ||w||²
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
BETA_OLS       = np.array([2.1, 1.4])
TILT_DEG       = 27.0
ASPECT         = 3.65
ELLIPSE_LEVELS = [0.35, 0.85, 1.8, 3.5]   # quadratic-form levels for RSS rings

N_FRAMES = 80
_lam_fwd = np.geomspace(6.0, 0.25, N_FRAMES // 2)
_lam_bwd = np.geomspace(0.25, 6.0, N_FRAMES // 2)
LAMBDAS  = np.concatenate([_lam_fwd, _lam_bwd])   # one full λ cycle

R_SCALE = 1.35   # constraint radius = R_SCALE / √λ  (so at λ=1, r≈1.35)

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE  (matches reference image)
# ══════════════════════════════════════════════════════════════════════════════
DISK_COLOR     = "#4472C4"   # solid blue
DISK_ALPHA     = 0.85
ELLIPSE_FILL   = "#c8e6c9"   # soft green
ELLIPSE_EDGE   = "#2e7d32"   # dark green
ELLIPSE_ALPHAS = [0.70, 0.58, 0.46, 0.34]   # innermost → outermost
DASH_COLOR     = "#E65C00"   # orange
AXIS_COLOR     = "black"
LABEL_COLOR    = "black"
LABEL_FS       = 11
AXIS_FS        = 14
TITLE_FS       = 13

# ══════════════════════════════════════════════════════════════════════════════
#  QUADRATIC FORM  A  for RSS ellipses:  (β − β_OLS)ᵀ A (β − β_OLS) = c
# ══════════════════════════════════════════════════════════════════════════════
tilt_rad           = np.radians(TILT_DEG)
c_t, s_t           = np.cos(tilt_rad), np.sin(tilt_rad)
R_rot              = np.array([[c_t, -s_t], [s_t, c_t]])
A                  = R_rot @ np.diag([1.0 / ASPECT**2, 1.0]) @ R_rot.T
eig_vals, eig_vecs = np.linalg.eigh(A)
patch_angle        = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))


def ellipse_wh(c):
    """Width and height (diameters) of the RSS ellipse at level c."""
    return 2.0 * np.sqrt(c / eig_vals[0]), 2.0 * np.sqrt(c / eig_vals[1])


def ridge_estimate(r):
    """Ridge estimate: OLS when constraint is inactive, else tangency point on disk boundary."""
    if np.linalg.norm(BETA_OLS) <= r:
        return BETA_OLS.copy()
    phis  = np.linspace(0, 2 * np.pi, 8_000)
    cands = r * np.stack([np.cos(phis), np.sin(phis)], axis=1)
    rss   = np.einsum("ni,ij,nj->n", cands - BETA_OLS, A, cands - BETA_OLS)
    phi_  = phis[np.argmin(rss)]
    return r * np.array([np.cos(phi_), np.sin(phi_)])


# ── Pre-compute per-frame data ────────────────────────────────────────────────
frame_data = [
    (lam, R_SCALE / np.sqrt(lam), ridge_estimate(R_SCALE / np.sqrt(lam)))
    for lam in LAMBDAS
]

# ── Fixed view limits (worst-case r_max + outermost ellipse) ──────────────────
r_max        = max(r for _, r, _ in frame_data)
w_out, h_out = ellipse_wh(ELLIPSE_LEVELS[-1])
x_lo = min(-r_max * 1.05, BETA_OLS[0] - w_out * 0.50) - 0.2
x_hi = max( r_max * 1.05, BETA_OLS[0] + w_out * 0.50) + 0.5
y_lo = min(-r_max * 1.05, BETA_OLS[1] - h_out * 0.50) - 0.2
y_hi = max( r_max * 1.05, BETA_OLS[1] + h_out * 0.50) + 0.4

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════
x_range = x_hi - x_lo
y_range = y_hi - y_lo
fig_w   = 9.0
fig_h   = fig_w * (y_range / x_range) / 0.88   # 0.88 = top−bottom margin fraction

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_aspect("equal")

ax.set_title(r"$L(w) = \|y - Xw\|^2 + \lambda\|w\|^2$", fontsize=TITLE_FS, pad=6)
plt.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.05)

# ── Coordinate axes ───────────────────────────────────────────────────────────
_ax_arr = dict(arrowstyle="-|>", color=AXIS_COLOR, lw=1.6, mutation_scale=12)
ax.annotate("", xy=(x_hi - 0.2, 0), xytext=(x_lo + 0.1, 0),
            arrowprops=_ax_arr, zorder=7)
ax.annotate("", xy=(0, y_hi - 0.15), xytext=(0, y_lo + 0.1),
            arrowprops=_ax_arr, zorder=7)
ax.text(x_hi - 0.12, -0.18, r"$w_1$", fontsize=AXIS_FS,
        ha="left", va="top", style="italic")
ax.text(-0.12, y_hi - 0.15, r"$w_2$", fontsize=AXIS_FS,
        ha="right", va="center", style="italic")

# ── Static RSS ellipses (draw outer → inner so innermost renders on top) ──────
for lvl, alpha in zip(reversed(ELLIPSE_LEVELS), reversed(ELLIPSE_ALPHAS)):
    w, h = ellipse_wh(lvl)
    ax.add_patch(Ellipse(
        xy=BETA_OLS, width=w, height=h, angle=patch_angle,
        facecolor=ELLIPSE_FILL, alpha=alpha,
        edgecolor=ELLIPSE_EDGE, linewidth=1.3, zorder=5,
    ))

# ── OLS point + label (static) ───────────────────────────────────────────────
ax.scatter(*BETA_OLS, color="black", s=55, zorder=11)
ax.annotate(
    "OLS estimate",
    xy=BETA_OLS,
    xytext=BETA_OLS + np.array([0.25, 0.55]),
    fontsize=LABEL_FS, color=LABEL_COLOR, fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=LABEL_COLOR, lw=0.9, mutation_scale=10),
    ha="left", va="bottom", zorder=12,
)

# ── Animated artists ──────────────────────────────────────────────────────────
disk_patch = plt.Circle((0, 0), 1.0, facecolor=DISK_COLOR, alpha=DISK_ALPHA,
                         edgecolor="none", zorder=3)
ax.add_patch(disk_patch)

(dash_line,)  = ax.plot([], [], color=DASH_COLOR, lw=1.6,
                         linestyle=(0, (6, 4)), zorder=6)
ridge_dot     = ax.scatter([], [], color="black", s=55, zorder=11)
ridge_txt     = ax.text(0, 0, "Ridge estimate", fontsize=LABEL_FS,
                         color=LABEL_COLOR, fontweight="bold", ha="right", va="bottom", zorder=12)
(ridge_conn,) = ax.plot([], [], color=LABEL_COLOR, lw=0.9, zorder=12)
lam_txt       = ax.text(x_lo + 0.15, y_hi - 0.15, "",
                         fontsize=11, ha="left", va="top", zorder=13)


def update(i):
    lam, r, br = frame_data[i]

    disk_patch.set_radius(np.linalg.norm(br))
    dash_line.set_data([BETA_OLS[0], br[0]], [BETA_OLS[1], br[1]])
    ridge_dot.set_offsets([br])

    rtxt = br + np.array([-0.9, 0.65])
    ridge_txt.set_position(rtxt)
    ridge_conn.set_data([rtxt[0], br[0]], [rtxt[1], br[1]])

    lam_txt.set_text(f"λ = {lam:.2f}")


anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=80)

stamp    = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
out_path = OUTPUT_DIR / f"{stamp}_ridge_constraint_diagram.gif"
anim.save(str(out_path), writer=animation.PillowWriter(fps=15))
print(f"Saved to: {out_path}")
plt.show()
