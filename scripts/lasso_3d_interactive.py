"""
lasso_3d_interactive.py

Interactive 3-D viewer for the Lasso (L1) constraint against the
least-squares loss ellipsoid.

  Slider  — adjust λ; the octahedron and loss ellipsoids update live.
  Mouse   — click-drag to rotate, scroll to zoom (view is preserved on update).

  Blue octahedron   — L1 ball  |w0| + |w1| + |w2| ≤ λ
  Orange ellipsoids — nested level-sets of  L(w) = ||y - Xw||²
  Orange dot        — constrained minimum
  White star        — w_optimal (unconstrained, outside the ball)
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Loss: quadratic bowl with minimum outside the L1 ball ─────────────────────
W_OPT   = np.array([2.1, 1.7, 1.4])
A, B, C = 0.60, 1.00, 0.80
L1_OPT  = float(np.sum(np.abs(W_OPT)))   # = 5.2; beyond this λ, no constraint

def loss(w0, w1, w2):
    return ((w0 - W_OPT[0])**2 / A
          + (w1 - W_OPT[1])**2 / B
          + (w2 - W_OPT[2])**2 / C)

# ── Geometry helpers ──────────────────────────────────────────────────────────
def octa_faces(t):
    v = np.array([
        [ t, 0, 0], [-t, 0, 0],
        [ 0, t, 0], [ 0,-t, 0],
        [ 0, 0, t], [ 0, 0,-t],
    ])
    tris = [[0,2,4],[0,4,3],[0,3,5],[0,5,2],
            [1,4,2],[1,2,5],[1,5,3],[1,3,4]]
    return [[v[i].tolist() for i in tri] for tri in tris]

def ellipsoid_surf(level, nu=60, nv=30):
    ra = np.sqrt(level * A);  rb = np.sqrt(level * B);  rc = np.sqrt(level * C)
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, nu), np.linspace(0, np.pi, nv))
    return (W_OPT[0] + ra * np.cos(u) * np.sin(v),
            W_OPT[1] + rb * np.sin(u) * np.sin(v),
            W_OPT[2] + rc * np.cos(v))

def constrained_min(t, n=120):
    if L1_OPT <= t:
        return W_OPT.copy(), 0.0
    best_L, best_w = np.inf, np.array([t, 0., 0.])
    for s in [(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),
              (-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]:
        u = np.linspace(0, t, n);  v = np.linspace(0, t, n)
        U, V = np.meshgrid(u, v);  m = (U + V) <= t
        w0s = s[0]*U[m];  w1s = s[1]*V[m];  w2s = s[2]*(t - U[m] - V[m])
        Ls  = loss(w0s, w1s, w2s);  idx = int(Ls.argmin())
        if Ls[idx] < best_L:
            best_L = Ls[idx]
            best_w = np.array([w0s[idx], w1s[idx], w2s[idx]])
    return best_w, best_L

# ── Precompute lookup table so the slider is responsive ───────────────────────
LAM_MIN, LAM_MAX, LAM_INIT = 0.10, 4.0, 1.2
N_PRE     = 120
lam_table = np.linspace(LAM_MIN, LAM_MAX, N_PRE)
cw_table  = np.zeros((N_PRE, 3))
cL_table  = np.zeros(N_PRE)

print("Precomputing constrained minima…")
for i, lam in enumerate(lam_table):
    cw_table[i], cL_table[i] = constrained_min(lam)
print("Done.")

def lookup(lam):
    lam = float(np.clip(lam, LAM_MIN, LAM_MAX))
    i   = max(0, min(int(np.searchsorted(lam_table, lam)) - 1, N_PRE - 2))
    f   = (lam - lam_table[i]) / (lam_table[i + 1] - lam_table[i])
    return ((1-f)*cw_table[i] + f*cw_table[i+1],
            float((1-f)*cL_table[i] + f*cL_table[i+1]))

# ── Figure layout ─────────────────────────────────────────────────────────────
LIM = max(LAM_MAX, float(W_OPT.max())) + 0.5

fig = plt.figure(figsize=(10, 9))
fig.patch.set_facecolor("#0e1117")

ax    = fig.add_axes([0.02, 0.12, 0.96, 0.85], projection="3d")
ax_sl = fig.add_axes([0.15, 0.045, 0.70, 0.025])

ax.view_init(elev=22, azim=-65)

slider = Slider(ax_sl, r"  $\lambda$", LAM_MIN, LAM_MAX,
                valinit=LAM_INIT, color="#2563eb", initcolor="none")
slider.label.set_color("white")
slider.label.set_fontsize(12)
slider.valtext.set_color("white")
ax_sl.set_facecolor("#1a1e2e")
for spine in ax_sl.spines.values():
    spine.set_edgecolor("#3a3a3a")

# ── Redraw ────────────────────────────────────────────────────────────────────
def redraw(lam):
    elev, azim = ax.elev, ax.azim   # preserve user's current rotation
    ax.cla()
    ax.set_facecolor("#0e1117")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#3a3a3a")

    c_w, c_L = lookup(lam)
    inactive  = (L1_OPT <= lam)

    # Octahedron
    ax.add_collection3d(Poly3DCollection(
        octa_faces(lam),
        alpha=0.38, facecolor="#2563eb", edgecolor="#93c5fd", linewidth=0.7,
    ))

    # Loss ellipsoids
    if not inactive and c_L > 0.02:
        for frac, alpha in [(0.30, 0.09), (0.62, 0.13), (1.00, 0.25)]:
            ax.plot_surface(*ellipsoid_surf(c_L * frac),
                            color="#f97316", alpha=alpha,
                            linewidth=0, antialiased=True)
    else:
        # Constraint inactive: draw a tiny reference ellipsoid at the optimum
        ax.plot_surface(*ellipsoid_surf(0.04),
                        color="#f97316", alpha=0.12,
                        linewidth=0, antialiased=True)

    # Unconstrained minimum
    ax.scatter(*W_OPT, color="white", s=180, marker="*", zorder=10,
               label=r"$w_{\mathrm{optimal}}$ (unconstrained)")

    # Constrained minimum
    ax.scatter(*c_w, color="#f97316", s=110,
               edgecolors="white", linewidths=0.9, zorder=10,
               label=rf"Constrained min  $\hat{{w}}={c_w.round(2)}$")

    # Dashed connector
    ax.plot([c_w[0], W_OPT[0]], [c_w[1], W_OPT[1]], [c_w[2], W_OPT[2]],
            color="white", lw=1.0, linestyle="--", alpha=0.45)

    inactive_note = "  (constraint inactive)" if inactive else ""
    ax.set_title(
        rf"Lasso (L1)  —  $|w_0|+|w_1|+|w_2| \leq \lambda={lam:.2f}${inactive_note}"
        "\n"
        r"$\mathcal{L}(w) = \|y - Xw\|^2$  (least squares)",
        color="white", fontsize=12, fontweight="bold", pad=12,
    )

    ax.set_xlim(-LIM, LIM);  ax.set_ylim(-LIM, LIM);  ax.set_zlim(-LIM, LIM)
    ax.set_xlabel("$w_0$", color="white", labelpad=8, fontsize=11)
    ax.set_ylabel("$w_1$", color="white", labelpad=8, fontsize=11)
    ax.set_zlabel("$w_2$", color="white", labelpad=8, fontsize=11)
    ax.tick_params(colors="white", labelsize=8)
    ax.legend(fontsize=8, loc="upper right", facecolor="#1a1e2e",
              edgecolor="#555", labelcolor="white", framealpha=0.85)

    ax.view_init(elev=elev, azim=azim)
    fig.canvas.draw_idle()

slider.on_changed(redraw)
redraw(LAM_INIT)
plt.show()
