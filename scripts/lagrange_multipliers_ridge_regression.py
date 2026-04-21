import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# ── Problem setup ──────────────────────────────────────────────────────────────
# Minimize  f(w) = ||Xw - b||^2  s.t.  ||w||^2 = k
#
# In 2-D weight space: f(w) = (w - w_ols)^T (w - w_ols)  [A = I for clean circles]
# Constrained optimum: project w_ols radially onto the constraint circle.
# ──────────────────────────────────────────────────────────────────────────────

w_ols = np.array([2.1, 1.1])
k     = 1.0

def loss(w1, w2):
    return (w1 - w_ols[0])**2 + (w2 - w_ols[1])**2

w_opt = w_ols / np.linalg.norm(w_ols) * np.sqrt(k)
z_opt = loss(*w_opt)

# ── Loss surface (paraboloid) ──────────────────────────────────────────────────
N = 70
w1_arr = np.linspace(-1.6, 2.8, N)
w2_arr = np.linspace(-1.6, 2.4, N)
W1, W2 = np.meshgrid(w1_arr, w2_arr)
Z = loss(W1, W2)

# ── Constraint geometry ────────────────────────────────────────────────────────
t = np.linspace(0, 2 * np.pi, 600)
circ_w1 = np.sqrt(k) * np.cos(t)
circ_w2 = np.sqrt(k) * np.sin(t)
circ_z  = loss(circ_w1, circ_w2)

legend_handles = [
    Patch(facecolor="#b07adb", alpha=0.75, edgecolor="white",
          label=r"Loss landscape $\ell(w) = \|Xw - b\|^2$"),
    Patch(facecolor="limegreen", alpha=0.7, edgecolor="black",
          label=r"Feasible region $\|w\|^2 \leq k$"),
    Line2D([0], [0], color="black", linewidth=2.5,
           label=r"Constraint boundary $\|w\|^2 = k$"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow",
           markeredgecolor="black", markersize=9,
           label=r"Constrained optimum $w^*$"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#e67e22",
           markeredgecolor="#e67e22", markersize=11,
           label=r"Unconstrained minimum $\hat{w}_\mathrm{OLS}$"),
]


def build_figure(figsize, title_y, math_y, subtitle_y, adjust_kw):
    """Build the full figure; return (fig, ax)."""
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Green filled disk at z = 0
    disk_verts = list(zip(circ_w1, circ_w2, np.zeros_like(t)))
    ax.add_collection3d(Poly3DCollection([disk_verts],
                                          facecolor="limegreen", alpha=0.55,
                                          edgecolor="none", zorder=1))

    # Black constraint circle at z = 0
    ax.plot(circ_w1, circ_w2, 0,
            color="black", linewidth=3.0, zorder=5, solid_capstyle="round")

    # Purple paraboloid bowl
    ax.plot_surface(W1, W2, Z,
                    color="#b07adb", alpha=0.62,
                    rstride=2, cstride=2,
                    linewidth=0.25, edgecolor="white",
                    zorder=2)

    # Constraint circle ON the bowl
    ax.plot(circ_w1, circ_w2, circ_z,
            color="black", linewidth=2.5, zorder=7,
            solid_capstyle="round")

    # Dashed curtain lines
    n_curtain = 24
    idx = np.linspace(0, len(t) - 1, n_curtain, dtype=int)
    for i in idx:
        ax.plot([circ_w1[i], circ_w1[i]],
                [circ_w2[i], circ_w2[i]],
                [0,          circ_z[i]],
                color="black", linewidth=0.5, alpha=0.25, zorder=3)

    # Constrained optimum — dot on bowl
    ax.scatter([w_opt[0]], [w_opt[1]], [z_opt],
               s=140, color="yellow", edgecolor="black", linewidth=1.5,
               zorder=10, depthshade=False)
    ax.scatter([w_opt[0]], [w_opt[1]], [0],
               s=100, color="yellow", edgecolor="black", linewidth=1.5,
               zorder=10, depthshade=False)
    ax.plot([w_opt[0], w_opt[0]], [w_opt[1], w_opt[1]], [0, z_opt],
            "k--", linewidth=1.3, alpha=0.85, zorder=6)
    ax.text(w_opt[0] + 0.18, w_opt[1] + 0.12, z_opt * 0.45,
            "constrained\noptimum", fontsize=10.5, color="black", zorder=11,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # OLS minimum
    ax.scatter([w_ols[0]], [w_ols[1]], [0],
               s=140, color="#e67e22", marker="*", zorder=10, depthshade=False)
    ax.text(w_ols[0] + 0.1, w_ols[1], 0.25,
            r"$\hat{w}_\mathrm{OLS}$", fontsize=11, color="#e67e22",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # Colored axis arrows
    ax_lim, z_top = 1.7, 22.0
    kw = dict(arrow_length_ratio=0.07, linewidth=2.8, zorder=9)
    ax.quiver(-ax_lim, 0, 0,  2 * ax_lim + 0.8, 0, 0, color="#27ae60", **kw)
    ax.quiver(0, -ax_lim, 0,  0, 2 * ax_lim + 0.5, 0, color="#2980b9", **kw)
    ax.quiver(0, 0, 0,         0, 0, z_top,          color="#c0392b",
              arrow_length_ratio=0.04, linewidth=2.8, zorder=9)

    def axis_label(x, y, z, txt, color):
        ax.text(x, y, z, txt, fontsize=14, color=color, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    axis_label(ax_lim + 0.3, 0,            0,      r"$w_1$",     "#27ae60")
    axis_label(0,            ax_lim + 0.25, 0,     r"$w_2$",     "#2980b9")
    axis_label(0.15,         0.15,         z_top,  r"$\ell(w)$", "#c0392b")

    ax.set_axis_off()

    # Title / subtitle
    fig.text(0.5, title_y,
             "L2 Ridge Regression expressed as Lagrange Multipliers",
             ha="center", va="top", fontsize=16, fontweight="bold")
    fig.text(0.5, math_y,
             r"$\min_w \|Xw - b\|^2 \;\text{s.t.}\; \|w\|^2 = k$",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, subtitle_y,
             r"$\mathcal{L}(w,\lambda)=\|Xw{-}b\|^2+\lambda(\|w\|^2{-}k)$"
             "          "
             r"Stationarity: $(X^TX + \lambda I)\,w = X^Tb$",
             ha="center", va="top", fontsize=11, color="#333333")

    # Legend — lower left
    ax.legend(handles=legend_handles, loc="lower left",
              bbox_to_anchor=(0.0, 0.0), fontsize=10,
              framealpha=0.92, edgecolor="#aaaaaa")

    fig.subplots_adjust(**adjust_kw)
    return fig, ax


def make_gif(fig, ax, out_path, n_frames=120, interval=60):
    ELEV, AZ_START = 24, -48

    def animate(frame):
        ax.view_init(elev=ELEV, azim=AZ_START + frame * (360 / n_frames))
        return []

    anim = FuncAnimation(fig, animate, frames=n_frames,
                         interval=interval, blit=False)
    print(f"Saving {n_frames}-frame GIF → {out_path}  (this may take ~30 s) …")
    anim.save(out_path, writer=PillowWriter(fps=1000 // interval))
    print("Done.")
    plt.close(fig)


# ── Original GIF ───────────────────────────────────────────────────────────────
fig, ax = build_figure(
    figsize=(11, 9),
    title_y=0.99, math_y=0.95, subtitle_y=0.91,
    adjust_kw=dict(top=0.88, bottom=0.04, left=0.04, right=0.96),
)
make_gif(fig, ax, "outputs/lagrange_ridge_regression_spin.gif")

# ── Thumbnail GIF — compact, tight margins ─────────────────────────────────────
fig, ax = build_figure(
    figsize=(8, 7),
    title_y=0.99, math_y=0.94, subtitle_y=0.89,
    adjust_kw=dict(top=0.97, bottom=0.01, left=0.01, right=0.99),
)
make_gif(fig, ax, "outputs/lagrange_ridge_regression_spin_thumbnail.gif")
