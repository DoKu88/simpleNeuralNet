import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

f = lambda x: x**2
grad_f = lambda x: 2 * x

x0 = -1.0
y = np.linspace(-2.5, 2.5, 400)

fx = f(y)
tangent = f(x0) + grad_f(x0) * (y - x0)

fig, ax = plt.subplots(figsize=(6, 5))

y_top = 7.0

# Shade epigraph (region above the curve)
ax.fill_between(y, fx, y_top, color="#a8d8ea", alpha=0.5, zorder=1)

# Tangent line
ax.plot(y, tangent, color="#e63946", linewidth=1.5, zorder=2)

# Convex function
ax.plot(y, fx, color="#1d3557", linewidth=2.5, zorder=3)

# Tangent point
ax.plot(x0, f(x0), "o", color="#e63946", markersize=7, zorder=4)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.5, y_top)
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.spines[["top", "right"]].set_visible(False)

# Minimal legend
curve_patch = mpatches.Patch(color="#1d3557", label=r"$f(y)$")
tangent_patch = mpatches.Patch(color="#e63946", label=r"$f(x) + \nabla f(x)^\top(y-x)$")
ax.legend(handles=[curve_patch, tangent_patch], frameon=True, fontsize=11, loc="upper left",
          facecolor="white", edgecolor="black")

plt.tight_layout()
plt.savefig("outputs/convex_underestimator.png", dpi=150, bbox_inches="tight")
plt.show()
