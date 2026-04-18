import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.patheffects as pe

# ── helpers ──────────────────────────────────────────────────────────────────

def proj(a, b):
    """Project vector a onto vector b."""
    return (np.dot(a, b) / np.dot(b, b)) * b


def angle_between(u, v):
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.clip(cos, -1, 1))


# ── Figure setup ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.size": 14,
})

fig = plt.figure(figsize=(13, 5.5), facecolor="white")
fig.suptitle("Vector Projection", fontsize=18, fontweight="bold", y=0.98)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — 2D vector-onto-vector  (replicates the diagram)
# ─────────────────────────────────────────────────────────────────────────────

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_aspect("equal")
ax1.set_xlim(-0.3, 3.6)
ax1.set_ylim(-0.3, 3.2)
ax1.axis("off")
ax1.set_title(r"Projection of $\vec{b}$ onto $\vec{a}$", fontsize=15, pad=10)

O = np.array([0.0, 0.0])
a = np.array([0.6, 2.8])   # blue vector ↑
b = np.array([3.0, 1.0])   # red vector  →

p = proj(a, b)              # foot of perpendicular (projection)
e = a - p                   # error / residual

# extend the b-line for context
b_hat = b / np.linalg.norm(b)
line_end = b_hat * 3.8

kw_arr = dict(length_includes_head=True, head_width=0.08, head_length=0.13)

# grey dashed span line
ax1.plot([0, line_end[0]], [0, line_end[1]], color="grey",
         linewidth=1.2, linestyle="--", zorder=0)

# vector b (blue)
ax1.annotate("", xy=a, xytext=O,
             arrowprops=dict(arrowstyle="-|>", color="#0072B2", lw=2.2,
                             mutation_scale=18))
ax1.text(a[0] - 0.22, a[1] + 0.08, r"$\vec{b}$",
         fontsize=17, color="#0072B2", fontweight="bold")

# vector a (red)
ax1.annotate("", xy=b, xytext=O,
             arrowprops=dict(arrowstyle="-|>", color="#D55E00", lw=2.2,
                             mutation_scale=18))
ax1.text(b[0] + 0.08, b[1] + 0.06, r"$\vec{a}$",
         fontsize=17, color="#D55E00", fontweight="bold")

# projection p (green, on the b-line)
ax1.annotate("", xy=p, xytext=O,
             arrowprops=dict(arrowstyle="-|>", color="#009E73", lw=2.0,
                             mutation_scale=15))
mid_p = p * 0.52
ax1.text(mid_p[0] + 0.06, mid_p[1] - 0.22, r"$p$",
         fontsize=16, color="#009E73", fontstyle="italic", fontweight="bold")

# error e (orange dashed, from p to b)
ax1.annotate("", xy=a, xytext=p,
             arrowprops=dict(arrowstyle="-|>", color="#332288", lw=1.8,
                             linestyle="dashed", mutation_scale=14))
mid_e = (p + a) * 0.5
ax1.text(mid_e[0] + 0.10, mid_e[1] + 0.05, r"$e = \vec{b} - p$",
         fontsize=14, color="#332288", fontstyle="italic", fontweight="bold")

# right-angle mark at p
perp_size = 0.15
b_unit = b_hat
e_unit = e / np.linalg.norm(e)
corner = p + perp_size * b_unit + perp_size * e_unit
sq = np.array([p + perp_size * b_unit,
               corner,
               p + perp_size * e_unit])
ax1.plot(sq[:, 0], sq[:, 1], color="grey", lw=1.0)


# axes lines
ax1.annotate("", xy=(3.5, 0), xytext=(-0.2, 0),
             arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12))
ax1.annotate("", xy=(0, 3.1), xytext=(0, -0.2),
             arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12))
ax1.text(3.55, -0.12, r"$x$", fontsize=15, fontweight="bold")
ax1.text(0.06, 3.12, r"$y$", fontsize=15, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — 3D: project a vector onto the xy-plane
# ─────────────────────────────────────────────────────────────────────────────

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.set_title("Projection onto the xy-plane (2D)", fontsize=15, pad=10)

v = np.array([2.0, 1.5, 2.5])   # the original 3D (or "1D concept") vector
v_proj = np.array([v[0], v[1], 0.0])   # shadow on xy-plane
v_perp = v - v_proj             # perpendicular component (z-axis)

def draw_arrow3d(ax, start, end, color, lw=2, mutation=15, label=None):
    """Draw a 3-D arrow via a quiver."""
    d = end - start
    ax.quiver(*start, *d, color=color, linewidth=lw, arrow_length_ratio=0.12,
              label=label)

# shaded xy-plane (unit square scaled)
span = 3.2
verts = [[(0, 0, 0), (span, 0, 0), (span, span, 0), (0, span, 0)]]
plane = Poly3DCollection(verts, alpha=0.15, facecolor="steelblue",
                         edgecolor="steelblue", linewidth=0.8)
ax2.add_collection3d(plane)

O3 = np.zeros(3)

# original vector v (dark red)
draw_arrow3d(ax2, O3, v, color="#D55E00", lw=2.5, label=r"$\vec{b}$")
ax2.text(*v * 1.04, r"$\vec{b}$", color="#D55E00", fontsize=16, fontweight="bold")

# projection onto plane (green)
draw_arrow3d(ax2, O3, v_proj, color="#009E73", lw=2.2,
             label=r"$p = \mathrm{proj}_{xy}(\vec{v})$")
ax2.text(v_proj[0] + 0.05, v_proj[1] + 0.05, -0.15,
         r"$p$", color="#009E73", fontsize=16, fontstyle="italic", fontweight="bold")

# perpendicular component (pink dashed)
draw_arrow3d(ax2, v_proj, v, color="#332288", lw=1.8)
ax2.text(v[0] + 0.05, v[1], v[2] * 0.55,
         r"$e$", color="#332288", fontsize=15, fontstyle="italic", fontweight="bold")

# dashed drop line from tip of v to its shadow
ax2.plot([v[0], v[0]], [v[1], v[1]], [0, v[2]],
         color="grey", linestyle=":", linewidth=1.2)

# axes labels
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
ax2.set_xticklabels([]); ax2.set_yticklabels([]); ax2.set_zticklabels([])
ax2.set_xlim(0, span); ax2.set_ylim(0, span); ax2.set_zlim(0, span)
ax2.set_box_aspect([1, 1, 1])
ax2.view_init(elev=22, azim=-55)

leg = ax2.legend(loc="upper left", fontsize=13, framealpha=0.7)


plt.tight_layout()
plt.savefig("outputs/vector_projection.png", dpi=150, bbox_inches="tight")
plt.show()
