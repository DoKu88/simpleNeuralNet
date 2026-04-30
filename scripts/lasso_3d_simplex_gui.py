"""
lasso_3d_simplex_gui.py

Side-by-side interactive 3-D viewer for the Lasso (L1) constraint.

  Left panel  — full L1 octahedron  |w0|+|w1|+|w2| ≤ λ
  Right panel — Q1 simplex face only (w0,w1,w2 ≥ 0), camera starts on the
                (1,1,1) ray so the triangular face is seen head-on

  Slider — adjust λ; both panels update simultaneously
  Mouse  — rotate/zoom each panel independently; angles are preserved
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Patch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
LAM_MIN, LAM_MAX, LAM_INIT = 0.0, L1_OPT, 1.2
N_PRE     = 120
lam_table = np.linspace(LAM_MIN, LAM_MAX, N_PRE)
cw_full   = np.zeros((N_PRE, 3));  cL_full  = np.zeros(N_PRE)
cw_simp   = np.zeros((N_PRE, 3));  cL_simp  = np.zeros(N_PRE)

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

# ── Default camera angles ──────────────────────────────────────────────────────
# Full octahedron: classic tilted view
ELEV_OCT, AZIM_OCT = 25.0, -30.0

# Q1 simplex: looking from the (1,1,1) ray toward the origin.
# elevation = arcsin(1/sqrt(3)), azimuth = 45°
ELEV_SIM = float(np.degrees(np.arcsin(1.0 / np.sqrt(3))))   # ≈ 35.26°
AZIM_SIM = 45.0

# ── Figure layout ──────────────────────────────────────────────────────────────
LIM_OCT = max(LAM_MAX, float(W_OPT.max())) + 0.6
LIM_SIM = LAM_MAX + 0.5

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor("#0e1117")

ax_oct = fig.add_axes([-0.06, 0.10, 0.62, 0.76], projection="3d")
ax_sim = fig.add_axes([0.44, 0.10, 0.62, 0.76], projection="3d")
ax_sl  = fig.add_axes([0.20, 0.03, 0.60, 0.025])

fig.suptitle(
    r"L1 Lasso Regression — $\mathcal{L}(w,\lambda)"
    r"= \|y - Xw\|^2 + \lambda\sum_i|w_i|$",
    color="white", fontsize=14, fontweight="bold", y=0.97,
)

ax_oct.view_init(elev=ELEV_OCT, azim=AZIM_OCT)
ax_sim.view_init(elev=ELEV_SIM, azim=AZIM_SIM)

# Slider
slider = Slider(ax_sl, r"  $\lambda$", LAM_MIN, LAM_MAX,
                valinit=LAM_INIT, color="#2563eb", initcolor="none")
slider.label.set_color("white");  slider.label.set_fontsize(12)
slider.valtext.set_color("white")
ax_sl.set_facecolor("#1a1e2e")
for sp in ax_sl.spines.values():
    sp.set_edgecolor("#3a3a3a")

# ── Shared axis styling ────────────────────────────────────────────────────────
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

# ── Redraw ─────────────────────────────────────────────────────────────────────
def redraw(lam):
    # Save each panel's current rotation independently
    elev_o, azim_o = ax_oct.elev, ax_oct.azim
    elev_s, azim_s = ax_sim.elev, ax_sim.azim

    cw_o, cL_o = lookup(lam, cw_full, cL_full)
    cw_s, cL_s = lookup(lam, cw_simp, cL_simp)

    # ── Left: full octahedron ─────────────────────────────────────────────────
    ax_oct.cla()
    style_ax(ax_oct, -LIM_OCT, LIM_OCT)
    if lam > 1e-6:
        ax_oct.add_collection3d(Poly3DCollection(
            octa_faces(lam),
            alpha=0.38, facecolor="#2563eb", edgecolor="#93c5fd", linewidth=0.7,
        ))
    draw_points(ax_oct, cw_o,
                r"L1 constraint  $\sum_i|w_i|\leq\lambda$")
    ax_oct.set_title(
        rf"(a)  Full Octahedron — $\sum_i|w_i| \leq \lambda={lam:.2f}$",
        color="white", fontsize=11, fontweight="bold", pad=-30,
    )
    ax_oct.view_init(elev=elev_o, azim=azim_o)

    # ── Right: Q1 simplex ─────────────────────────────────────────────────────
    ax_sim.cla()
    style_ax(ax_sim, 0.0, LIM_SIM)
    if lam > 1e-6:
        ax_sim.add_collection3d(Poly3DCollection(
            simplex_face(lam),
            alpha=0.48, facecolor="#2563eb", edgecolor="#93c5fd", linewidth=1.8,
        ))
        # Triangle perimeter
        ring = np.array([[lam,0,0],[0,lam,0],[0,0,lam],[lam,0,0]])
        ax_sim.plot(ring[:,0], ring[:,1], ring[:,2], color="#93c5fd", lw=1.8)
        # Dotted spokes to origin
        for pt in [[lam,0,0],[0,lam,0],[0,0,lam]]:
            ax_sim.plot([0,pt[0]],[0,pt[1]],[0,pt[2]],
                        color="#93c5fd", lw=0.8, linestyle=":", alpha=0.55)
    draw_points(ax_sim, cw_s,
                r"L1 constraint  $\sum_i w_i\leq\lambda,\ w_i\geq0$")
    ax_sim.set_title(
        rf"(b)  Q1 Simplex — $\sum_i w_i \leq \lambda={lam:.2f},\ w_i\geq0$",
        color="white", fontsize=11, fontweight="bold", pad=-30,
    )
    ax_sim.view_init(elev=elev_s, azim=azim_s)

    fig.canvas.draw_idle()

slider.on_changed(redraw)
redraw(LAM_INIT)
plt.show()
