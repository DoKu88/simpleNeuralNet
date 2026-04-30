"""
l1_norm_explorer.py

Interactive explorer for the L1 norm  Σ|wᵢ|  in 1–3 dimensions.
Three panels shown simultaneously left-to-right: 1-D → 2-D → 3-D.
Slider — adjust the highlighted level λ (updates all panels at once).

  1-D  f(w₁) = |w₁|                         V-shape with λ level marked
  2-D  f(w₁,w₂) = |w₁|+|w₂|               pyramid surface + λ diamond slice
  3-D  |w₁|+|w₂|+|w₃| = λ                  octahedron level set
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Constants ─────────────────────────────────────────────────────────────────
LAM_MIN, LAM_MAX, LAM_INIT = 0.10, 3.0, 1.5
LIM       = 3.2
BG        = "#0e1117"
BLUE      = "#2563eb"
BLUE_LITE = "#93c5fd"
ORANGE    = "#f97316"

# ── Precomputed static geometry ───────────────────────────────────────────────
_g2 = np.linspace(-LIM, LIM, 80)
_W1G, _W2G = np.meshgrid(_g2, _g2)
_Z2        = np.abs(_W1G) + np.abs(_W2G)          # pyramid surface (static)
_wv        = np.linspace(-LIM, LIM, 800)
_fv        = np.abs(_wv)

def octa_faces(t):
    v = np.array([[t,0,0],[-t,0,0],[0,t,0],[0,-t,0],[0,0,t],[0,0,-t]])
    f = [[0,2,4],[0,4,3],[0,3,5],[0,5,2],
         [1,4,2],[1,2,5],[1,5,3],[1,3,4]]
    return [[v[i].tolist() for i in tri] for tri in f]

def diamond_xy(t):
    return np.array([[t,0],[0,t],[-t,0],[0,-t],[t,0]])

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 7.5))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(bottom=0.10, left=0.08, right=0.98,
                    top=0.93, wspace=0.08)

# ── Slider ────────────────────────────────────────────────────────────────────
ax_sl = fig.add_axes([0.15, 0.03, 0.70, 0.025])
ax_sl.set_facecolor("#1a1e2e")
for sp in ax_sl.spines.values():
    sp.set_edgecolor("#3a3a3a")
slider = Slider(ax_sl, r"  $\lambda$", LAM_MIN, LAM_MAX,
                valinit=LAM_INIT, color=BLUE, initcolor="none")
slider.label.set_color("white");  slider.label.set_fontsize(12)
slider.valtext.set_color("white")

# ── Panel 1 — 1-D ─────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_facecolor(BG)
for sp in ax1.spines.values():
    sp.set_edgecolor("#3a3a3a")
ax1.tick_params(colors="white", labelsize=9)

ax1.plot(_wv, _fv, color=BLUE_LITE, lw=2.2, label=r"$f(w_1)=|w_1|$")
ax1.axhline(0, color="#3a3a3a", lw=0.5)
ax1.axvline(0, color="#3a3a3a", lw=0.5)
ax1.set_xlim(-LIM, LIM);  ax1.set_ylim(-0.15, LIM + 0.3)
ax1.set_xlabel("$w_1$", color="white", fontsize=12)
ax1.set_ylabel(r"$\sum_i|w_i|$", color="white", fontsize=12)
ax1.set_title(r"1-D:  $f(w_1)=|w_1|$",
              color="white", fontsize=13, fontweight="bold")

_hline1 = ax1.axhline(LAM_INIT, color=ORANGE, lw=1.8, ls="--",
                       label=rf"$\lambda={LAM_INIT:.2f}$")
_scat1  = ax1.scatter([-LAM_INIT, LAM_INIT], [LAM_INIT, LAM_INIT],
                       color=ORANGE, s=90, zorder=5)
_fill1  = [ax1.fill_between(_wv, _fv, LAM_INIT, where=(_fv <= LAM_INIT),
                              alpha=0.20, color=BLUE,
                              label=r"$|w_1|\leq\lambda$")]
_leg1   = ax1.legend(fontsize=10, facecolor="#1a1e2e", edgecolor="#555",
                     labelcolor="white", framealpha=0.85)

# Shrink ax1 height so it matches the visual weight of the 3-D panels
_p1 = ax1.get_position()
ax1.set_position([_p1.x0, _p1.y0 + _p1.height * 0.12,
                  _p1.width, _p1.height * 0.78])

# ── Panel 2 — 2-D ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.set_facecolor(BG)
for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
    pane.fill = False;  pane.set_edgecolor("#3a3a3a")
ax2.tick_params(colors="white", labelsize=8)

ax2.plot_surface(_W1G, _W2G, np.minimum(_Z2, LIM),
                 cmap="Blues", alpha=0.55, linewidth=0, antialiased=True)
ax2.set_xlabel("$w_1$", color="white", labelpad=8, fontsize=11)
ax2.set_ylabel("$w_2$", color="white", labelpad=8, fontsize=11)
ax2.set_zlabel(r"$\sum_i|w_i|$", color="white", labelpad=8, fontsize=11)
ax2.set_xlim(-LIM, LIM);  ax2.set_ylim(-LIM, LIM);  ax2.set_zlim(0, LIM)
ax2.set_title(r"2-D:  $f(w_1,w_2)=|w_1|+|w_2|$",
              color="white", fontsize=12, fontweight="bold")
ax2.view_init(elev=22, azim=-60)

_d0     = diamond_xy(LAM_INIT)
_dline2, = ax2.plot(_d0[:,0], _d0[:,1], np.full(5, LAM_INIT),
                    color=ORANGE, lw=2.5, zorder=8,
                    label=rf"$\lambda={LAM_INIT:.2f}$")
_dscat2_ref = [ax2.scatter(_d0[:-1,0], _d0[:-1,1], np.full(4, LAM_INIT),
                            color=ORANGE, s=60, zorder=9)]
_leg2   = ax2.legend(fontsize=9, loc="upper right", facecolor="#1a1e2e",
                     edgecolor="#555", labelcolor="white", framealpha=0.85)

# ── Panel 3 — 3-D ─────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.set_facecolor(BG)
for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
    pane.fill = False;  pane.set_edgecolor("#3a3a3a")
ax3.tick_params(colors="white", labelsize=8)

ax3.set_xlabel("$w_1$", color="white", labelpad=8, fontsize=11)
ax3.set_ylabel("$w_2$", color="white", labelpad=8, fontsize=11)
ax3.set_zlabel("$w_3$", color="white", labelpad=8, fontsize=11)
ax3.set_xlim(-LIM, LIM);  ax3.set_ylim(-LIM, LIM);  ax3.set_zlim(-LIM, LIM)
ax3.set_title(r"3-D:  $|w_1|+|w_2|+|w_3|=\lambda$  (octahedron)",
              color="white", fontsize=12, fontweight="bold")
ax3.view_init(elev=22, azim=-60)

_octa_ref = [None]

def _refresh_octa(lam):
    if _octa_ref[0] is not None:
        _octa_ref[0].remove()
    coll = Poly3DCollection(octa_faces(lam), alpha=0.40,
                            facecolor=BLUE, edgecolor=BLUE_LITE, linewidth=0.8)
    ax3.add_collection3d(coll)
    _octa_ref[0] = coll

_refresh_octa(LAM_INIT)

# ── Update callback ───────────────────────────────────────────────────────────
def update(lam):
    # 1-D
    _hline1.set_ydata([lam, lam])
    _scat1.set_offsets(np.c_[[-lam, lam], [lam, lam]])
    if _fill1[0] is not None:
        _fill1[0].remove()
    _fill1[0] = ax1.fill_between(_wv, _fv, lam, where=(_fv <= lam),
                                  alpha=0.20, color=BLUE)
    _hline1.set_label(rf"$\lambda={lam:.2f}$")
    ax1.legend(fontsize=10, facecolor="#1a1e2e", edgecolor="#555",
               labelcolor="white", framealpha=0.85)

    # 2-D
    d = diamond_xy(lam)
    _dline2.set_data(d[:,0], d[:,1])
    _dline2.set_3d_properties(np.full(5, lam))
    _dline2.set_label(rf"$\lambda={lam:.2f}$")
    if _dscat2_ref[0] is not None:
        _dscat2_ref[0].remove()
    _dscat2_ref[0] = ax2.scatter(d[:-1,0], d[:-1,1], np.full(4, lam),
                                  color=ORANGE, s=60, zorder=9)
    ax2.legend(fontsize=9, loc="upper right", facecolor="#1a1e2e",
               edgecolor="#555", labelcolor="white", framealpha=0.85)

    # 3-D
    _refresh_octa(lam)
    ax3.set_title(
        r"3-D:  $|w_1|+|w_2|+|w_3|=\lambda$  (octahedron)"
        "\n" + rf"$\lambda={lam:.2f}$",
        color="white", fontsize=12, fontweight="bold")

    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()
