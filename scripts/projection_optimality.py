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
a  = np.array([4.0, 0.0])
b  = np.array([2.0, 3.0])

p       = (np.dot(b, a) / np.dot(a, a)) * a
e       = b - p
p_prime = p + np.array([-0.7, 0.0])
e_prime = b - p_prime

# ── helpers ───────────────────────────────────────────────────────────────────

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

# ── draw function ─────────────────────────────────────────────────────────────

def draw_diagram(show_e=True):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="white")
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 5.0)
    ax.set_ylim(-0.4, 3.4)
    ax.axis("off")

    # axes
    ax.plot([-0.2, 4.8], [0, 0], color="black", lw=1.2, zorder=0)
    arrow(ax, [-0.2, 0], [4.8, 0], "black", lw=1.2, ms=12)
    ax.plot([0, 0], [-0.3, 3.3], color="black", lw=1.2, zorder=0)
    arrow(ax, [0, -0.3], [0, 3.3], "black", lw=1.2, ms=12)
    ax.text(4.85, -0.12, r"$x$", fontsize=15)
    ax.text(0.07,  3.30, r"$y$", fontsize=15)

    # vector a
    arrow(ax, O, a, C_A, lw=2.5)
    ax.text(a[0] + 0.08, a[1] + 0.07, r"$\vec{a}$", fontsize=17, color=C_A)

    # vector b
    arrow(ax, O, b, C_B, lw=2.5)
    ax.text(b[0] - 0.30, b[1] + 0.10, r"$\vec{b}$", fontsize=17, color=C_B)

    # phi arc
    phi_r = 0.42
    phi_angle = np.degrees(np.arctan2(b[1], b[0]))
    ax.add_patch(mpatches.Arc(O, 2 * phi_r, 2 * phi_r,
                              angle=0, theta1=0, theta2=phi_angle,
                              color=C_B, lw=1.4))
    phi_mid_rad = np.radians(phi_angle / 2)
    ax.text((phi_r + 0.10) * np.cos(phi_mid_rad),
            (phi_r + 0.10) * np.sin(phi_mid_rad),
            r"$\phi$", fontsize=13, color=C_B)

    # optimal p and e
    if show_e:
        ax.plot(*p, 'o', color=C_OPT, ms=7, zorder=5)
        ax.text(p[0] + 0.05, -0.28, r"$p$", fontsize=15, color=C_OPT)
        arrow(ax, p, b, C_OPT, lw=2.0)
        mid_e = (p + b) * 0.5
        ax.text(mid_e[0] + 0.08, mid_e[1], r"$e$", fontsize=15, color=C_OPT)
        right_angle(ax, p, e, a)

    # suboptimal p' and e'
    ax.plot(*p_prime, 'o', color=C_SUB, ms=7, zorder=5)
    ax.text(p_prime[0] - 0.05, -0.28, r"$p'$", fontsize=15, color=C_SUB, ha="right")
    arrow(ax, p_prime, b, C_SUB, lw=2.0, ls="dashed")
    mid_ep  = (p_prime + b) * 0.5
    e_dir   = e_prime / np.linalg.norm(e_prime)
    right_perp = np.array([e_dir[1], -e_dir[0]])
    label_ep = mid_ep + 0.22 * right_perp
    ax.text(label_ep[0], label_ep[1], r"$e'$",
            fontsize=15, color=C_SUB, ha="center", va="center")

    # theta arc at p'
    theta_r     = 0.35
    e_prime_angle = np.degrees(np.arctan2(e_prime[1], e_prime[0]))
    ax.add_patch(mpatches.Arc(p_prime, 2 * theta_r, 2 * theta_r,
                              angle=0, theta1=e_prime_angle, theta2=180,
                              color=C_SUB, lw=1.4))
    arc_mid_rad = np.radians((e_prime_angle + 180) / 2)
    ax.text(p_prime[0] + (theta_r + 0.08) * np.cos(arc_mid_rad),
            p_prime[1] + (theta_r + 0.08) * np.sin(arc_mid_rad),
            r"$\theta$", fontsize=13, color=C_SUB)

    # dotted triangle
    if show_e:
        tri_x = [p_prime[0], p[0], b[0], p_prime[0]]
        tri_y = [p_prime[1], p[1], b[1], p_prime[1]]
        ax.plot(tri_x, tri_y, color=C_GREY, lw=1.0, ls=":", zorder=2)

    plt.tight_layout()
    return fig

# ── output ────────────────────────────────────────────────────────────────────

fig1 = draw_diagram(show_e=False)
fig1.savefig("outputs/projection_suboptimal.png", dpi=150, bbox_inches="tight")

fig2 = draw_diagram(show_e=True)
fig2.savefig("outputs/projection_optimality.png", dpi=150, bbox_inches="tight")

plt.show()
