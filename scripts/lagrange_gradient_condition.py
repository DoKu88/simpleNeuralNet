import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Setup ──────────────────────────────────────────────────────────────────────
# f(x) = ||x - x_free||^2   (bowl, minimum at x_free)
# g(x) = ||x||^2 = k        (circular constraint, radius r)
#
# Constrained optimum x0: point on circle closest to x_free.
# Non-optimal x1: another point on the circle where ∇f is NOT parallel to ∇g.
# ──────────────────────────────────────────────────────────────────────────────

x_free = np.array([3.2, 1.6])   # unconstrained minimum of f
r      = 2.2                     # constraint radius  (||x||^2 = k = r^2)

def f(x, y):       return (x - x_free[0])**2 + (y - x_free[1])**2
def grad_f(p):     return 2 * (p - x_free)
def grad_g(p):     return 2 * p                  # ∇(||x||^2)
def tangent(p):                                  # unit tangent to circle at p
    t = np.array([-p[1], p[0]])
    return t / np.linalg.norm(t)

# Constrained optimum (projection of x_free onto circle)
x0 = x_free / np.linalg.norm(x_free) * r

# Non-optimal: rotate ~115° from x0 on the circle
phi = np.arctan2(x0[1], x0[0]) + np.radians(115)
x1  = np.array([r * np.cos(phi), r * np.sin(phi)])

# ── Grid for contours ──────────────────────────────────────────────────────────
xs = np.linspace(-3.2, 4.8, 400)
ys = np.linspace(-3.2, 3.8, 400)
X, Y = np.meshgrid(xs, ys)
F    = f(X, Y)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafafa")

# Filled contour (loss landscape)
lvl_fill = np.linspace(0, 28, 60)
cf = ax.contourf(X, Y, F, levels=lvl_fill, cmap="Blues", alpha=0.35, zorder=0)

# Contour lines
lvl_line = [0.3, 0.8, 2, 4, 7, 11, 16, 22]
ax.contour(X, Y, F, levels=lvl_line, colors="#2060a8", linewidths=1.1,
           alpha=0.65, zorder=1)

# Constraint circle
theta = np.linspace(0, 2 * np.pi, 600)
ax.plot(r * np.cos(theta), r * np.sin(theta),
        color="black", linewidth=2.8, zorder=3,
        label=r"$g(x)=c$:  $\|x\|^2 = k$")

# Unconstrained minimum
ax.plot(*x_free, "*", color="#e67e22", markersize=15, zorder=8,
        label=r"Unconstrained min $x^*_\mathrm{free}$")

# ── Helper: draw a gradient arrow ─────────────────────────────────────────────
def draw_arrow(ax, base, vec, color, lw=2.5, ls="-", alpha=1.0, zorder=6):
    ax.annotate("",
                xy=base + vec, xytext=base,
                arrowprops=dict(arrowstyle="-|>",
                                color=color, lw=lw,
                                linestyle=ls,
                                mutation_scale=18),
                zorder=zorder, alpha=alpha)

def label(ax, pos, txt, color, dx=0.0, dy=0.0, fs=12, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(pos[0] + dx, pos[1] + dy, txt, color=color,
            fontsize=fs, fontweight=fw,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            zorder=12)

SCALE_OPT  = 0.72   # arrow display length at x0
SCALE_NOPT = 1.4    # arrow display length at x1

# ══════════════════════════════════════════════════════════════════════════════
#  PANEL A — Constrained optimum x0
# ══════════════════════════════════════════════════════════════════════════════
gf0 = grad_f(x0);  gf0_u = gf0 / np.linalg.norm(gf0) * SCALE_OPT
gg0 = grad_g(x0);  gg0_u = gg0 / np.linalg.norm(gg0) * SCALE_OPT
tan0 = tangent(x0)

ax.plot(*x0, "o", color="#27ae60", markersize=13, zorder=9,
        label=r"$x_0$ : constrained optimum")

# Tangent line at x0
tlen = 1.1
ax.plot([x0[0] - tlen*tan0[0], x0[0] + tlen*tan0[0]],
        [x0[1] - tlen*tan0[1], x0[1] + tlen*tan0[1]],
        "--", color="#555555", linewidth=1.6, alpha=0.7, zorder=4)
label(ax, x0 + tlen*tan0, "tangent to $g=c$",
      "#555555", dx=0.05, dy=0.05, fs=10)

# ∇f and ∇g arrows (overlapping = parallel)
draw_arrow(ax, x0, gf0_u, "#1a6faf", lw=3.0)
draw_arrow(ax, x0, gg0_u, "#c0392b", lw=3.0)

gf0_perp = np.array([-gf0_u[1], gf0_u[0]]) / np.linalg.norm(gf0_u) * 0.22
label(ax, x0 + gf0_u + gf0_perp, r"$\nabla f(x_0)$", "#1a6faf", dx=0.0, dy=0.0, bold=True)
label(ax, x0 + gg0_u, r"$\nabla g(x_0)$", "#c0392b", dx=-0.1, dy=0.08, bold=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL B — Non-optimal point x1
# ══════════════════════════════════════════════════════════════════════════════
gf1 = grad_f(x1);  gf1_u = gf1 / np.linalg.norm(gf1) * SCALE_NOPT
gg1 = grad_g(x1);  gg1_u = gg1 / np.linalg.norm(gg1) * SCALE_NOPT
tan1 = tangent(x1)

# Decompose ∇f into tangential + normal components
gf1_tang_scalar = np.dot(gf1_u, tan1)
gf1_norm_scalar = np.dot(gf1_u, x1 / np.linalg.norm(x1))

# Match magnitude of tangential vec to the improvement arrow (SCALE_NOPT)
gf1_tang_vec = np.sign(gf1_tang_scalar) * tan1 * SCALE_NOPT
gf1_norm_vec = gf1_norm_scalar * (x1 / np.linalg.norm(x1))

ax.plot(*x1, "s", color="#e67e22", markersize=13, zorder=9,
        label=r"$x_1$ : non-optimal point on $g=c$")

# ∇f full arrow
draw_arrow(ax, x1, gf1_u, "#1a6faf", lw=3.5, alpha=1.0)
gf1_perp = np.array([-gf1_u[1], gf1_u[0]]) / np.linalg.norm(gf1_u) * 0.22
label(ax, x1 + gf1_u + gf1_perp, r"$\nabla f(x_1)$", "#1a6faf", dx=0.0, dy=0.0, bold=True)

# ∇g arrow
draw_arrow(ax, x1, gg1_u, "#c0392b", lw=3.5, alpha=1.0)
label(ax, x1 + gg1_u, r"$\nabla g(x_1)$", "#c0392b", dx=-0.1, dy=0.08, bold=True)

# Tangential component (purple dashed)
draw_arrow(ax, x1, gf1_tang_vec, "#8e44ad", lw=2.8, ls="dashed", zorder=7)
label(ax, x1 + gf1_tang_vec,
      r"$\nabla f_{\!\parallel}$" + "\n(tangential)",
      "#8e44ad", dx=0.06, dy=-0.35, fs=10)


# Right-angle tick between tangential and normal components
perp_scale = 0.13
v1 = tan1 * perp_scale
v2 = (x1 / np.linalg.norm(x1)) * perp_scale
corner = x1 + v1 + v2
ax.plot([x1[0]+v1[0], corner[0], x1[0]+v2[0]],
        [x1[1]+v1[1], corner[1], x1[1]+v2[1]],
        color="#8e44ad", linewidth=1.2, alpha=0.7, zorder=8)

# Straight arrow showing direction of improvement along constraint (opposite to ∇f_‖)
improve_vec = -np.sign(gf1_tang_scalar) * tan1 * SCALE_NOPT
draw_arrow(ax, x1, improve_vec, "#27ae60", lw=4.5, zorder=5)



# ── Labels, axes, legend ───────────────────────────────────────────────────────
ax.set_xlim(-3.2, 4.8)
ax.set_ylim(-3.2, 3.8)
ax.set_aspect("equal")
ax.axhline(0, color="k", linewidth=0.4, alpha=0.35)
ax.axvline(0, color="k", linewidth=0.4, alpha=0.35)
ax.set_xlabel(r"$x_1$", fontsize=13)
ax.set_ylabel(r"$x_2$", fontsize=13)
ax.set_title(
    r"Lagrange Multiplier Condition: $\nabla f(x_0) = \lambda\,\nabla g(x_0)$"
    "\n"
    r"At a constrained local extremum, gradients of $f$ and $g$ are parallel",
    fontsize=13, pad=10)

# Custom legend
legend_extras = [
    Line2D([0],[0], color="#1a6faf", linewidth=2.5,
           label=r"$\nabla f$ (gradient of objective)"),
    Line2D([0],[0], color="#c0392b", linewidth=2.5,
           label=r"$\nabla g$ (gradient of constraint)"),
    Line2D([0],[0], color="#8e44ad", linewidth=2.0, linestyle="--",
           label=r"$\nabla f_{\!\parallel}$ (tangential component of $\nabla f$)"),
    Line2D([0],[0], color="#27ae60", linewidth=3.0,
           label=r"Direction of improvement along $g=c$"),
]
handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(handles=handles + legend_extras,
          labels=labels_leg + [h.get_label() for h in legend_extras],
          fontsize=10, loc="lower left", framealpha=0.92, edgecolor="#aaaaaa")

cb = plt.colorbar(cf, ax=ax, fraction=0.03, pad=0.02, label=r"$f(x)$")
cb.set_ticks(range(0, 29, 4))
plt.tight_layout()
plt.savefig("outputs/lagrange_gradient_condition.png", dpi=150, bbox_inches="tight")
plt.show()
