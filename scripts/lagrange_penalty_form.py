import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ── Setup ──────────────────────────────────────────────────────────────────────
x_f = np.array([3.0, 2.2])      # f-minimum

def f(x1, x2):    return (x1 - x_f[0])**2 + (x2 - x_f[1])**2
def g(x1, x2):    return x1**2 + x2**2
def x_star(l):    return x_f / (1.0 + l)

xs = np.linspace(-0.5, 3.8, 500)
ys = np.linspace(-0.5, 3.2, 500)
X, Y = np.meshgrid(xs, ys)
F, G = f(X, Y), g(X, Y)

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── f contours (background) ────────────────────────────────────────────────────
cf = ax.contourf(X, Y, F, levels=np.linspace(0, 22, 50),
                 cmap="Blues", alpha=0.25, zorder=0)
ax.contour(X, Y, F, levels=[1, 3, 6, 10, 15, 21],
           colors="#2060a8", linewidths=1.1, alpha=0.5, zorder=1)

# ── Regularization path  x*(λ) = x_f / (1+λ) ─────────────────────────────────
lam_path = np.linspace(0, 30, 400)
path = np.stack([x_star(l) for l in lam_path])
ax.plot(path[:, 0], path[:, 1], color="black", linewidth=2.0,
        linestyle="--", alpha=0.6, zorder=4)

# ── λ dots + corresponding g-level-set circles ────────────────────────────────
lambdas = [0.5, 2.0, 6.0]
colors  = ["#8e44ad", "#e67e22", "#c0392b"]
theta   = np.linspace(0, 2 * np.pi, 400)

for lam, col in zip(lambdas, colors):
    xs_ = x_star(lam)
    radius = np.linalg.norm(xs_)          # g(x*(λ)) = radius^2

    # Thin circle: g(x) = ||x*(λ)||^2
    ax.plot(radius * np.cos(theta), radius * np.sin(theta),
            color=col, linewidth=1.2, alpha=0.8, zorder=3)

    # Optimal point dot
    ax.plot(*xs_, "o", color=col, markersize=12, zorder=8,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # λ label
    ax.text(xs_[0] + 0.09, xs_[1] + 0.10, fr"$\lambda={lam}$",
            color=col, fontsize=11, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=10)

# ── f-minimum (λ=0) ───────────────────────────────────────────────────────────
ax.plot(*x_f, "o", color="#27ae60", markersize=14, zorder=9,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")])
ax.text(x_f[0] + 0.08, x_f[1] + 0.12, r"$x^*_\mathrm{free}$  ($\lambda=0$)",
        color="#27ae60", fontsize=11, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=10)

# ── g-minimum (origin, λ→∞) ───────────────────────────────────────────────────
ax.plot(0, 0, "o", color="#c0392b", markersize=14, zorder=9,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")])
ax.text(0.08, -0.22, r"$0$  ($\lambda\to\infty$)",
        color="#c0392b", fontsize=11, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=10)

# ── Function definitions (top-left) ───────────────────────────────────────────
ax.text(0.02, 0.98,
        r"$f(x) = \|x - x^*_\mathrm{free}\|^2$"
        "\n"
        r"$g(x) = \|x\|^2$",
        transform=ax.transAxes, fontsize=11.5, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#aaaaaa", alpha=0.93),
        zorder=12)

# ── Legend (lower right) ──────────────────────────────────────────────────────
# Column-first fill order: all lines in col 1, all minimum-point markers in col 2
legend_handles = (
    [Line2D([0], [0], color=col, linewidth=1.8) for col in colors] +
    [Line2D([0], [0], color=col, marker="o", markersize=11,
            markerfacecolor=col, markeredgecolor="none", lw=0) for col in colors]
)
legend_labels = (
    [fr"Level Set for $\lambda={lam}$" for lam in lambdas]
    + ["Minimum Point"] * len(lambdas)
)
ax.legend(handles=legend_handles, labels=legend_labels,
          ncol=2, fontsize=10, loc="lower right",
          framealpha=0.93, edgecolor="#aaaaaa")

# ── Axes ───────────────────────────────────────────────────────────────────────
cb = plt.colorbar(cf, ax=ax, fraction=0.03, pad=0.02, label=r"$f(x)$")
cb.set_ticks(range(0, 23, 4))

ax.set_xlim(-0.5, 3.8)
ax.set_ylim(-0.5, 3.2)
ax.set_aspect("equal")
ax.axhline(0, color="k", lw=0.4, alpha=0.3)
ax.axvline(0, color="k", lw=0.4, alpha=0.3)
ax.set_xlabel(r"$x_1$", fontsize=13)
ax.set_ylabel(r"$x_2$", fontsize=13)
ax.set_title(r"$h(x,\lambda) = f(x) + \lambda\,g(x)$   —   $\lambda$ hyperparameter",
             fontsize=12, pad=10)

plt.tight_layout()
plt.savefig("outputs/lagrange_penalty_form.png", dpi=150, bbox_inches="tight")
plt.show()
