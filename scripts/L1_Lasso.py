"""
L1_Lasso.py

Animated GIFs for Lasso (L1) regularisation.

  GIF 1 – 2-D: classic diamond constraint vs. elliptical loss contours.
           Diamond shrinks until it forces the optimum to a corner (sparsity).

  GIF 2 – 3-D: L1 ball (octahedron) in 3-D weight-space plus the ellipsoidal
           loss level-set that just touches it. The octahedron grows and
           shrinks (2 cycles) while the camera spins slowly (~270°).
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 – 3-D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# ── 2-D loss: tilted elliptical bowl (minimum outside the origin) ─────────────
W_OPT_2 = np.array([1.9, 1.6])
A2, B2  = 0.55, 1.15

def loss2d(w0, w1):
    return (w0 - W_OPT_2[0])**2 / A2 + (w1 - W_OPT_2[1])**2 / B2

# ── 3-D loss ───────────────────────────────────────────────────────────────────
W_OPT_3 = np.array([2.1, 1.7, 1.4])
A3, B3, C3 = 0.60, 1.00, 0.80

def loss3d_grid(w0, w1, w2):
    return ((w0 - W_OPT_3[0])**2 / A3
          + (w1 - W_OPT_3[1])**2 / B3
          + (w2 - W_OPT_3[2])**2 / C3)

# ── Diamond path (|w0|+|w1| = t) ─────────────────────────────────────────────
def diamond_pts(t, n=500):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r   = 1.0 / (np.abs(np.cos(ang)) + np.abs(np.sin(ang)) + 1e-14)
    return np.column_stack([t * r * np.cos(ang), t * r * np.sin(ang)])

# ── 2-D constrained minimum ───────────────────────────────────────────────────
def constrained_min_2d(t):
    """argmin_{|w0|+|w1|≤t} loss2d"""
    if abs(W_OPT_2[0]) + abs(W_OPT_2[1]) <= t:
        return W_OPT_2.copy()
    best_L, best_w = np.inf, np.array([t, 0.0])
    n = 5000
    for s0, s1 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        lam = np.linspace(0, t, n)
        Ls  = loss2d(s0 * lam, s1 * (t - lam))
        idx = int(Ls.argmin())
        if Ls[idx] < best_L:
            best_L = Ls[idx]
            best_w = np.array([s0 * lam[idx], s1 * (t - lam[idx])])
    return best_w

# ── Octahedron faces (|w0|+|w1|+|w2| = t) ────────────────────────────────────
def octa_faces(t):
    v = np.array([
        [ t, 0, 0], [-t, 0, 0],
        [ 0, t, 0], [ 0,-t, 0],
        [ 0, 0, t], [ 0, 0,-t],
    ])
    tris = [
        [0,2,4],[0,4,3],[0,3,5],[0,5,2],
        [1,4,2],[1,2,5],[1,5,3],[1,3,4],
    ]
    return [[v[i].tolist() for i in tri] for tri in tris]

# ── Ellipsoid surface for level-set loss3d(w) = level ─────────────────────────
def ellipsoid_surf(level, nu=42, nv=24):
    if level <= 0:
        return None
    ra = np.sqrt(level * A3)
    rb = np.sqrt(level * B3)
    rc = np.sqrt(level * C3)
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, nu),
                        np.linspace(0,   np.pi, nv))
    return (W_OPT_3[0] + ra * np.cos(u) * np.sin(v),
            W_OPT_3[1] + rb * np.sin(u) * np.sin(v),
            W_OPT_3[2] + rc * np.cos(v))

# ── 3-D constrained minimum ───────────────────────────────────────────────────
def constrained_min_3d(t, n=130):
    """Grid search over all 8 faces of the octahedron."""
    best_L, best_w = np.inf, np.array([t, 0., 0.])
    for s in [(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),
              (-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]:
        u = np.linspace(0, t, n)
        v = np.linspace(0, t, n)
        U, V = np.meshgrid(u, v)
        m    = (U + V) <= t
        w0s  =  s[0] * U[m]
        w1s  =  s[1] * V[m]
        w2s  =  s[2] * (t - U[m] - V[m])
        Ls   = loss3d_grid(w0s, w1s, w2s)
        idx  = int(Ls.argmin())
        if Ls[idx] < best_L:
            best_L = Ls[idx]
            best_w = np.array([w0s[idx], w1s[idx], w2s[idx]])
    return best_w, best_L


# =============================================================================
# GIF 1 – 2-D
# =============================================================================
LIM2 = 2.8
w0v  = np.linspace(-LIM2, LIM2, 300)
w1v  = np.linspace(-LIM2, LIM2, 300)
W0G, W1G = np.meshgrid(w0v, w1v)
LG2      = loss2d(W0G, W1G)

# Cosine wave: large → zero → large, seamless loop (last frame → first frame).
_N2 = 160
_phase = np.linspace(0, 2 * np.pi, _N2, endpoint=False)
T_SEQ  = 3.6 * (1 + np.cos(_phase)) / 2

fig2, ax2 = plt.subplots(figsize=(7, 7), facecolor="white")
ax2.set_facecolor("white")

lev2 = np.linspace(0.0, 16.0, 26)
ax2.contourf(W0G, W1G, LG2, levels=lev2, cmap="YlOrRd_r", alpha=0.70)
ax2.contour( W0G, W1G, LG2, levels=lev2[::2], colors="black",
             linewidths=0.4, alpha=0.22)

ax2.scatter([W_OPT_2[0]], [W_OPT_2[1]], color="black", s=160,
            marker="*", zorder=10, label=r"Unconstrained $w_{\mathrm{optimal}}$")
ax2.axhline(0, color="black", lw=0.4, alpha=0.25)
ax2.axvline(0, color="black", lw=0.4, alpha=0.25)

ax2.set_xlabel("$w_0$", fontsize=13, color="black")
ax2.set_ylabel("$w_1$", fontsize=13, color="black")
ax2.tick_params(colors="black", labelsize=9)
for sp in ax2.spines.values():
    sp.set_edgecolor("#aaa")
ax2.set_xlim(-LIM2, LIM2)
ax2.set_ylim(-LIM2, LIM2)
ax2.set_aspect("equal")
ax2.set_title(
    "Lasso (L1)\n"
    r"$\mathcal{L}(w) = \|y - Xw\|^2 + \lambda\sum_i|w_i|$",
    fontsize=13, fontweight="bold", color="black", pad=10,
)

dpoly = MplPolygon(
    diamond_pts(T_SEQ[0]), closed=True,
    facecolor="#2563eb", edgecolor="#1d4ed8",
    linewidth=2.0, alpha=0.22, zorder=4, label="L1 ball",
)
ax2.add_patch(dpoly)

cpt2, = ax2.plot([], [], "o", color="#dc2626", ms=11,
                 markeredgecolor="white", mew=0.8,
                 zorder=8, label="Constrained min")
tlbl2 = ax2.text(0.03, 0.04, rf"$\lambda = {T_SEQ[0]:.2f}$",
                 transform=ax2.transAxes, fontsize=12,
                 color="black", va="bottom")

ax2.legend(fontsize=9, loc="upper right", facecolor="white",
           edgecolor="#aaa", labelcolor="black", framealpha=0.90)


def anim2d(i):
    t = T_SEQ[i]
    dpoly.set_xy(diamond_pts(t))
    cw = constrained_min_2d(t)
    cpt2.set_data([cw[0]], [cw[1]])
    tlbl2.set_text(rf"$\lambda = {t:.2f}$")


ani2 = animation.FuncAnimation(fig2, anim2d, frames=len(T_SEQ), interval=50)
gif2 = OUTPUT_DIR / f"{stamp}_lasso_2d.gif"
ani2.save(gif2, writer="pillow", fps=20, dpi=130)
print(f"Saved: {gif2}")
plt.close(fig2)


# =============================================================================
# GIF 2 – 3-D  (octahedron grows/shrinks while camera spins slowly)
# =============================================================================
T_MIN_3D = 0.20
T_MAX_3D = 2.20
LIM3     = max(float(W_OPT_3.max()) + 0.65, T_MAX_3D + 0.4)

# Precompute constrained minima across the full t range so each frame is cheap.
N_PRECOMP  = 80
T_PRECOMP  = np.linspace(T_MIN_3D, T_MAX_3D, N_PRECOMP)
L1_NORM_OPT = float(np.sum(np.abs(W_OPT_3)))   # 2.1+1.7+1.4 = 5.2

print("Precomputing 3-D constrained minima …")
_cw_pre, _cL_pre = [], []
for _t in T_PRECOMP:
    if L1_NORM_OPT <= _t:          # unconstrained min is inside the ball
        _cw_pre.append(W_OPT_3.copy())
        _cL_pre.append(0.0)
    else:
        _cw, _cL = constrained_min_3d(_t)
        _cw_pre.append(_cw)
        _cL_pre.append(_cL)
PRECOMP_CW = np.array(_cw_pre)    # (N_PRECOMP, 3)
PRECOMP_CL = np.array(_cL_pre)    # (N_PRECOMP,)


def interp_constrained(t):
    """Linear interpolation into the precomputed table."""
    t  = float(np.clip(t, T_PRECOMP[0], T_PRECOMP[-1]))
    i  = int(np.searchsorted(T_PRECOMP, t)) - 1
    i  = max(0, min(i, N_PRECOMP - 2))
    f  = (t - T_PRECOMP[i]) / (T_PRECOMP[i + 1] - T_PRECOMP[i])
    cw = (1 - f) * PRECOMP_CW[i] + f * PRECOMP_CW[i + 1]
    cL = (1 - f) * PRECOMP_CL[i] + f * PRECOMP_CL[i + 1]
    return cw, float(cL)


# Animation sequences: 2 full grow-shrink cycles, camera drifts ~270°
N_FRAMES_3D = 240
phase    = np.linspace(0, 4 * np.pi, N_FRAMES_3D, endpoint=False)   # 2 cycles
T_SEQ_3D = T_MIN_3D + (T_MAX_3D - T_MIN_3D) * (1 - np.cos(phase)) / 2
AZIM_SEQ = -65.0 + np.linspace(0, 270.0, N_FRAMES_3D)

fig3 = plt.figure(figsize=(9, 8), facecolor="#0e1117")
ax3  = fig3.add_subplot(111, projection="3d")
fig3.suptitle(
    "Lasso (L1) — 3-D octahedron + loss level-sets\n"
    r"$\min_w\,\mathcal{L}(w)$  s.t.  $|w_0|+|w_1|+|w_2|\leq t$",
    fontsize=12, fontweight="bold", color="white", y=0.99,
)


def draw_3d(frame):
    ax3.cla()
    ax3.set_facecolor("#0e1117")
    for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#3a3a3a")

    t    = T_SEQ_3D[frame]
    azim = AZIM_SEQ[frame]
    c_w3, c_L3 = interp_constrained(t)

    ax3.view_init(elev=22, azim=azim % 360)

    # Octahedron — size follows t
    pc = Poly3DCollection(
        octa_faces(t),
        alpha=0.45, facecolor="#4C9BE8",
        edgecolor="#4CE8E8", linewidth=0.7,
    )
    ax3.add_collection3d(pc)

    # Ellipsoidal level-set that just touches the current octahedron.
    # Fade out when the ellipsoid grows huge (small t → high loss).
    if c_L3 > 0.02:
        alpha_e = float(np.clip(0.30 * min(1.0, 3.5 / c_L3), 0.05, 0.32))
        esurf   = ellipsoid_surf(c_L3)
        if esurf is not None:
            ax3.plot_surface(*esurf, color="#E87B4C", alpha=alpha_e,
                             linewidth=0, antialiased=True)

    # Unconstrained minimum (fixed star)
    ax3.scatter(*W_OPT_3, color="white", s=130, marker="*", zorder=10)

    # Constrained minimum (moves with t)
    ax3.scatter(*c_w3, color="#E87B4C", s=90,
                edgecolors="white", linewidths=0.9, zorder=10)

    # Dashed line: constrained → unconstrained
    ax3.plot([c_w3[0], W_OPT_3[0]],
             [c_w3[1], W_OPT_3[1]],
             [c_w3[2], W_OPT_3[2]],
             color="white", lw=0.9, linestyle="--", alpha=0.40)

    ax3.set_xlim(-LIM3, LIM3)
    ax3.set_ylim(-LIM3, LIM3)
    ax3.set_zlim(-LIM3, LIM3)
    ax3.set_xlabel("$w_0$", color="white", labelpad=6, fontsize=10)
    ax3.set_ylabel("$w_1$", color="white", labelpad=6, fontsize=10)
    ax3.set_zlabel("$w_2$", color="white", labelpad=6, fontsize=10)
    ax3.tick_params(colors="white", labelsize=7)

    ax3.text2D(0.03, 0.04, f"$t = {t:.2f}$",
               transform=ax3.transAxes, fontsize=11,
               color="white", va="bottom")


ani3 = animation.FuncAnimation(fig3, draw_3d, frames=N_FRAMES_3D, interval=55)
gif3 = OUTPUT_DIR / f"{stamp}_lasso_3d.gif"
ani3.save(gif3, writer="pillow", fps=18, dpi=100)
print(f"Saved: {gif3}")
plt.close(fig3)
