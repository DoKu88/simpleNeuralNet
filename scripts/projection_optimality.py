import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Wong colorblind-safe palette
C_A    = "#D55E00"   # vermillion  — vector a
C_B    = "#0072B2"   # blue        — vector b
C_OPT  = "#009E73"   # bluish-green — optimal p, e
C_SUB  = "#332288"   # indigo      — suboptimal p', e'
C_GREY = "#999999"

plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.size": 14,
})

# ── geometry ─────────────────────────────────────────────────────────────────

O  = np.array([0.0, 0.0])
a  = np.array([4.0, 0.0])          # horizontal — the projection target
b  = np.array([1.6, 2.4])          # the vector being projected

# optimal projection (foot of perpendicular)
p  = (np.dot(b, a) / np.dot(a, a)) * a   # = (1.6, 0)
e  = b - p                                 # perpendicular residual

# suboptimal projection — shifted left along a
p_prime = p + np.array([-1.2, 0.0])       # = (0.4, 0)
e_prime = b - p_prime                      # non-perpendicular residual

# ── figure ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
ax.set_aspect("equal")
ax.set_xlim(-0.4, 5.0)
ax.set_ylim(-0.5, 3.2)
ax.axis("off")
fig.suptitle("Why Orthogonal Projection Minimises Error",
             fontsize=16, fontweight="bold", y=0.97)

def arrow(ax, start, end, color, lw=2.0, ls="solid", ms=16, zorder=3):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, linestyle=ls, mutation_scale=ms),
                zorder=zorder)

def right_angle(ax, corner, v1, v2, size=0.13, color=C_GREY):
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    sq = np.array([corner + size * u1,
                   corner + size * u1 + size * u2,
                   corner + size * u2])
    ax.plot(sq[:, 0], sq[:, 1], color=color, lw=1.2, zorder=4)

# ── span / axis lines ────────────────────────────────────────────────────────

ax.plot([-0.2, 4.8], [0, 0], color="black", lw=1.2, zorder=0)   # x-axis
arrow(ax, [-0.2, 0], [4.8, 0], "black", lw=1.2, ms=12)
ax.plot([0, 0], [-0.3, 3.0], color="black", lw=1.2, zorder=0)   # y-axis
arrow(ax, [0, -0.3], [0, 3.0], "black", lw=1.2, ms=12)
ax.text(4.85, -0.12, r"$x$", fontsize=15)
ax.text(0.07,  3.05, r"$y$", fontsize=15)

# ── vector a ─────────────────────────────────────────────────────────────────

arrow(ax, O, a, C_A, lw=2.5)
ax.text(a[0] + 0.08, a[1] + 0.07, r"$\vec{a}$",
        fontsize=17, color=C_A)

# ── vector b ─────────────────────────────────────────────────────────────────

arrow(ax, O, b, C_B, lw=2.5)
ax.text(b[0] - 0.30, b[1] + 0.10, r"$\vec{b}$",
        fontsize=17, color=C_B)

# ── optimal projection p & error e ───────────────────────────────────────────

# p dot on x-axis
ax.plot(*p, 'o', color=C_OPT, ms=7, zorder=5)
ax.text(p[0] + 0.05, -0.28, r"$p$", fontsize=15, color=C_OPT)

# e: from p to tip of b  (perpendicular)
arrow(ax, p, b, C_OPT, lw=2.0)
mid_e = (p + b) * 0.5
ax.text(mid_e[0] + 0.08, mid_e[1], r"$e$",
        fontsize=15, color=C_OPT)

# right-angle mark at p
right_angle(ax, p, e, a)

# ── suboptimal projection p' & error e' ──────────────────────────────────────

# p' dot on x-axis
ax.plot(*p_prime, 'o', color=C_SUB, ms=7, zorder=5)
ax.text(p_prime[0] - 0.05, -0.28, r"$p'$", fontsize=15, color=C_SUB,
        ha="right")

# e': from p' to tip of b  (not perpendicular)
arrow(ax, p_prime, b, C_SUB, lw=2.0, ls="dashed")
mid_ep = (p_prime + b) * 0.5
ax.text(mid_ep[0] - 0.32, mid_ep[1] + 0.05, r"$e'$",
        fontsize=15, color=C_SUB)

# ── triangle p'–p–tip(b): highlight the right triangle ───────────────────────

tri_x = [p_prime[0], p[0], b[0], p_prime[0]]
tri_y = [p_prime[1], p[1], b[1], p_prime[1]]
ax.plot(tri_x, tri_y, color=C_GREY, lw=1.0, ls=":", zorder=2)

# |p - p'| brace label below x-axis
ax.annotate("", xy=(p[0], -0.42), xytext=(p_prime[0], -0.42),
            arrowprops=dict(arrowstyle="<->", color=C_GREY, lw=1.2))
ax.text((p[0] + p_prime[0]) * 0.5, -0.50,
        r"$|p - p'|$", fontsize=12, color=C_GREY, ha="center")

# ── law-of-cosines annotation ────────────────────────────────────────────────

formula = (
    r"$\|e'\|^2 = \|e\|^2 + \|p - p'\|^2$"
    "\n"
    r"$\therefore\;\|e'\|^2 \geq \|e\|^2$"
    "\n"
    r"equality iff $e \perp \vec{a}$"
)
ax.text(2.6, 2.55, formula, fontsize=12, color="black",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f5f5f5", ec=C_GREY, alpha=0.9),
        linespacing=1.6)

plt.tight_layout()
plt.savefig("outputs/projection_optimality.png", dpi=150, bbox_inches="tight")
plt.show()
