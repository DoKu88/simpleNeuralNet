"""
lasso_3d_boomerang.py

Animated GIF — side-by-side L1 Lasso viewer (same layout as lasso_3d_simplex_gui.py).
λ sweeps 0 → max → 0 in a seamless boomerang loop.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# ── Loss setup ─────────────────────────────────────────────────────────────────
W_OPT   = np.array([2.1, 1.7, 1.4])
A, B, C = 0.60, 1.00, 0.80
L1_OPT  = float(np.sum(np.abs(W_OPT)))   # = 5.2

def loss(w0, w1, w2):
    return ((w0 - W_OPT[0])**2 / A
          + (w1 - W_OPT[1])**2 / B
          + (w2 - W_OPT[2])**2 / C)

# ── Geometry helpers ───────────────────────────────────────────────────────────
def octa_faces(t):
    v = np.array([
        [ t, 0, 0], [-t, 0, 0],
        [ 0, t, 0], [ 0,-t, 0],
        [ 0, 0, t], [ 0, 0,-t],
    ])
    tris = [[0,2,4],[0,4,3],[0,3,5],[0,5,2],
            [1,4,2],[1,2,5],[1,5,3],[1,3,4]]
    return [[v[i].tolist() for i in tri] for tri in tris]

def simplex_face(t):
    return [[[t, 0, 0], [0, t, 0], [0, 0, t]]]

def constrained_min(t, signs, n=120):
    if L1_OPT <= t:
        return W_OPT.copy(), 0.0
    best_L, best_w = np.inf, np.array([t, 0., 0.])
    for s in signs:
        u = np.linspace(0, t, n);  v = np.linspace(0, t, n)
        U, V = np.meshgrid(u, v);  m = (U + V) <= t
        w0s = s[0]*U[m];  w1s = s[1]*V[m];  w2s = s[2]*(t - U[m] - V[m])
        Ls  = loss(w0s, w1s, w2s);  idx = int(Ls.argmin())
        if Ls[idx] < best_L:
            best_L = Ls[idx]
            best_w = np.array([w0s[idx], w1s[idx], w2s[idx]])
    return best_w, best_L

ALL_SIGNS = [(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),
             (-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]
Q1_SIGNS  = [(1, 1, 1)]

# ── Precompute ─────────────────────────────────────────────────────────────────
LAM_MIN, LAM_MAX = 0.0, L1_OPT
N_PRE     = 120
lam_table = np.linspace(LAM_MIN, LAM_MAX, N_PRE)
cw_full   = np.zeros((N_PRE, 3));  cL_full = np.zeros(N_PRE)
cw_simp   = np.zeros((N_PRE, 3));  cL_simp = np.zeros(N_PRE)

print("Precomputing constrained minima…")
for i, lam in enumerate(lam_table):
    cw_full[i], cL_full[i] = constrained_min(lam, ALL_SIGNS)
    cw_simp[i], cL_simp[i] = constrained_min(lam, Q1_SIGNS)
print("Done.")

def lookup(lam, cw_t, cL_t):
    lam = float(np.clip(lam, LAM_MIN, LAM_MAX))
    i   = max(0, min(int(np.searchsorted(lam_table, lam)) - 1, N_PRE - 2))
    f   = (lam - lam_table[i]) / (lam_table[i+1] - lam_table[i])
    return ((1-f)*cw_t[i] + f*cw_t[i+1],
            float((1-f)*cL_t[i] + f*cL_t[i+1]))

# ── λ sequence: pause@0 → ramp up → pause@max → ramp down (seamless) ─────────
N_HALF       = 50    # frames for each half-sweep
PAUSE_FRAMES = 12    # frames to hold at each endpoint

_up   = np.linspace(0, np.pi, N_HALF, endpoint=False)
_down = np.linspace(np.pi, 2 * np.pi, N_HALF, endpoint=False)
T_SEQ = np.concatenate([
    np.zeros(PAUSE_FRAMES),                          # hold at λ=0
    LAM_MAX * (1 - np.cos(_up))   / 2,              # ramp up to λ_max
    np.full(PAUSE_FRAMES, LAM_MAX),                  # hold at λ_max
    LAM_MAX * (1 - np.cos(_down)) / 2,              # ramp down to λ≈0
])

# ── Camera angles (match lasso_3d_simplex_gui.py) ─────────────────────────────
ELEV_OCT, AZIM_OCT = 25.0, -30.0
ELEV_SIM = float(np.degrees(np.arcsin(1.0 / np.sqrt(3))))  # ≈ 35.26°
AZIM_SIM = 45.0

# ── Figure layout ──────────────────────────────────────────────────────────────
LIM_OCT = max(LAM_MAX, float(W_OPT.max())) + 0.6
LIM_SIM = LAM_MAX + 0.5

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor("#0e1117")

ax_oct = fig.add_axes([-0.06, 0.10, 0.62, 0.76], projection="3d")
ax_sim = fig.add_axes([0.44, 0.10, 0.62, 0.76], projection="3d")
ax_sl  = fig.add_axes([0.20, 0.03, 0.60, 0.025])
ax_sl.set_facecolor("#1a1e2e")
for sp in ax_sl.spines.values():
    sp.set_edgecolor("#3a3a3a")

fig.suptitle(
    r"L1 Lasso Regression — $\mathcal{L}(w,\lambda)"
    r"= \|y - Xw\|^2 + \lambda\sum_i|w_i|$",
    color="white", fontsize=14, fontweight="bold", y=0.97,
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def style_ax(ax, lim_lo, lim_hi):
    ax.set_facecolor("#0e1117")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#3a3a3a")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_zlim(lim_lo, lim_hi)
    ax.set_xlabel("$w_0$", color="white", labelpad=6, fontsize=11)
    ax.set_ylabel("$w_1$", color="white", labelpad=6, fontsize=11)
    ax.set_zlabel("$w_2$", color="white", labelpad=6, fontsize=11)
    ax.tick_params(colors="white", labelsize=7)

def draw_points(ax, c_w, constraint_label):
    ax.scatter(*W_OPT, color="white", s=160, marker="*", zorder=10)
    ax.scatter(*c_w, color="#f97316", s=100,
               edgecolors="white", linewidths=0.9, zorder=10)
    ax.plot([c_w[0], W_OPT[0]], [c_w[1], W_OPT[1]], [c_w[2], W_OPT[2]],
            color="white", lw=1.2, linestyle="--", alpha=0.80)
    h_blue = Patch(facecolor="#2563eb", edgecolor="#93c5fd",
                   alpha=0.6, label=constraint_label)
    h_line, = ax.plot([], [], [], color="white", lw=1.2, linestyle="--",
                      alpha=0.80, label=r"$\|w_{\mathrm{opt}} - \hat{w}\|$ (gap to constraint)")
    h_opt,  = ax.plot([], [], [], "*", color="white", ms=9,
                      label=r"$w_{\mathrm{optimal}}$ (unconstrained min)")
    h_cw,   = ax.plot([], [], [], "o", color="#f97316", ms=7,
                      markeredgecolor="white", markeredgewidth=0.8,
                      label=rf"$\hat{{w}}$ (constrained min) $= [{c_w[0]:.2f},\ {c_w[1]:.2f},\ {c_w[2]:.2f}]$")
    ax.legend(handles=[h_blue, h_line, h_opt, h_cw],
              fontsize=8, loc="lower right", facecolor="#1a1e2e",
              edgecolor="#555", labelcolor="white", framealpha=0.85)

# ── Animation ──────────────────────────────────────────────────────────────────
def animate(frame_idx):
    lam      = T_SEQ[frame_idx]
    cw_o, _  = lookup(lam, cw_full, cL_full)
    cw_s, _  = lookup(lam, cw_simp, cL_simp)

    ax_oct.cla()
    style_ax(ax_oct, -LIM_OCT, LIM_OCT)
    if lam > 1e-6:
        ax_oct.add_collection3d(Poly3DCollection(
            octa_faces(lam),
            alpha=0.38, facecolor="#2563eb", edgecolor="#93c5fd", linewidth=0.7,
        ))
    draw_points(ax_oct, cw_o, r"L1 constraint  $\sum_i|w_i|\leq\lambda$")
    ax_oct.set_title(
        rf"(a)  Full Octahedron — $\sum_i|w_i| \leq \lambda={lam:.2f}$",
        color="white", fontsize=11, fontweight="bold", pad=-30,
    )
    ax_oct.view_init(elev=ELEV_OCT, azim=AZIM_OCT)

    ax_sim.cla()
    style_ax(ax_sim, 0.0, LIM_SIM)
    if lam > 1e-6:
        ax_sim.add_collection3d(Poly3DCollection(
            simplex_face(lam),
            alpha=0.48, facecolor="#2563eb", edgecolor="#93c5fd", linewidth=1.8,
        ))
        ring = np.array([[lam,0,0],[0,lam,0],[0,0,lam],[lam,0,0]])
        ax_sim.plot(ring[:,0], ring[:,1], ring[:,2], color="#93c5fd", lw=1.8)
        for pt in [[lam,0,0],[0,lam,0],[0,0,lam]]:
            ax_sim.plot([0,pt[0]],[0,pt[1]],[0,pt[2]],
                        color="#93c5fd", lw=0.8, linestyle=":", alpha=0.55)
    draw_points(ax_sim, cw_s, r"L1 constraint  $\sum_i w_i\leq\lambda,\ w_i\geq0$")
    ax_sim.set_title(
        rf"(b)  Q1 Simplex — $\sum_i w_i \leq \lambda={lam:.2f},\ w_i\geq0$",
        color="white", fontsize=11, fontweight="bold", pad=-30,
    )
    ax_sim.view_init(elev=ELEV_SIM, azim=AZIM_SIM)

    # ── Lambda progress bar ───────────────────────────────────────────────────
    ax_sl.cla()
    ax_sl.set_facecolor("#1a1e2e")
    ax_sl.set_xlim(0, LAM_MAX)
    ax_sl.set_ylim(0, 1)
    ax_sl.set_xticks([])
    ax_sl.set_yticks([])
    for sp in ax_sl.spines.values():
        sp.set_edgecolor("#3a3a3a")
    ax_sl.fill_betweenx([0, 1], 0, lam, color="#2563eb")
    ax_sl.axvline(lam, color="white", lw=2, alpha=0.9)
    ax_sl.text(-0.02, 0.5, r"$\lambda$", ha="right", va="center",
               color="white", fontsize=12, transform=ax_sl.transAxes)
    ax_sl.text(1.02, 0.5, f"{lam:.2f}", ha="left", va="center",
               color="white", fontsize=10, transform=ax_sl.transAxes)

    if frame_idx % 10 == 0:
        print(f"  frame {frame_idx+1}/{len(T_SEQ)}  λ={lam:.2f}")


print("Rendering frames…")
ani = animation.FuncAnimation(fig, animate, frames=len(T_SEQ), interval=60)
out_path = OUTPUT_DIR / f"{stamp}_lasso_3d_boomerang.gif"
ani.save(out_path, writer="pillow", fps=15, dpi=100)
print(f"Saved: {out_path}")
plt.close(fig)
