import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# ── colours (same palette as vc_dim_visualization.py) ────────────────────────
RED   = "#E74C3C"
BLUE  = "#3498DB"
BG    = "#000000"
PANEL = "#000000"
GOLD  = "#FFFFFF"
WHITE = "#EAEAEA"
GREEN = "#2ECC71"
GRAY  = "#7F8C8D"

# ── dataset ───────────────────────────────────────────────────────────────────
np.random.seed(7)

N = 20
# Spread points along the line y=x by sampling x uniformly, then
# offset perpendicularly so half fall above, half below
x1 = np.random.uniform(-2, 2, N)
offsets = np.random.uniform(0.15, 1.2, N)          # distance from the line
signs   = np.array([1, -1] * (N // 2))             # exactly half above, half below
np.random.shuffle(signs)
x2 = x1 + signs * offsets                          # y = x ± offset
x2 = np.clip(x2, -2.4, 2.4)
pts = np.column_stack([x1, x2])

# Ground truth: +1 if above y=x, -1 if below
labels_true = np.where(x2 > x1, 1, -1)

# Flip ~10% of labels as noise
noise_idx = np.random.choice(N, size=max(1, int(0.10 * N)), replace=False)
labels = labels_true.copy()
labels[noise_idx] *= -1

# ── fit models ────────────────────────────────────────────────────────────────
# Linear classifier (VC dim = 3 in R²)
clf_linear = LogisticRegression(C=1e6, max_iter=5000)
clf_linear.fit(pts, labels)

# Degree-20 polynomial classifier (VC dim = 21 for 1-D threshold polynomial)
clf_poly = Pipeline([
    ("poly", PolynomialFeatures(degree=20, include_bias=True)),
    ("clf",  LogisticRegression(C=1e6, max_iter=10000)),
])
clf_poly.fit(pts, labels)

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7), facecolor=BG)
fig.patch.set_facecolor(BG)

fig.text(
    0.5, 0.96,
    "VC Dimension — Underfitting vs Overfitting",
    ha="center", va="top", fontsize=20, fontweight="bold",
    color=GOLD, fontfamily="monospace",
)

ax_l = fig.add_axes([0.04, 0.10, 0.44, 0.76], facecolor=PANEL)
ax_r = fig.add_axes([0.52, 0.10, 0.44, 0.76], facecolor=PANEL)

LIM = 2.5

def setup_ax(ax, title):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.set_title(title, color=WHITE, fontsize=12, pad=10, fontfamily="monospace")

setup_ax(ax_l, "VC Dimension - Linear Model = 3")
setup_ax(ax_r, "VC Dimension - Polynomial (deg 20) = 21")

# ── decision regions + boundary ───────────────────────────────────────────────
def draw_regions(ax, clf):
    xs = np.linspace(-LIM, LIM, 400)
    ys = np.linspace(-LIM, LIM, 400)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    ZZ = clf.predict(grid).reshape(XX.shape).astype(float)

    ax.contourf(XX, YY, ZZ, levels=[-2, 0, 2],
                colors=[BLUE, RED], alpha=0.20, zorder=1)
    ax.contour(XX, YY, ZZ, levels=[0],
               colors=[GREEN], linewidths=2.5, zorder=3)

draw_regions(ax_l, clf_linear)
draw_regions(ax_r, clf_poly)

# ── true boundary (y = x) ─────────────────────────────────────────────────────
for ax in (ax_l, ax_r):
    ax.plot([-LIM, LIM], [-LIM, LIM],
            color=GOLD, linewidth=1.5, linestyle="--",
            alpha=0.55, zorder=2)

# ── data points ───────────────────────────────────────────────────────────────
for ax in (ax_l, ax_r):
    for p, lbl in zip(pts, labels):
        color = RED if lbl == 1 else BLUE
        ax.scatter(*p, s=200, color=color, zorder=5,
                   edgecolors=WHITE, linewidths=1.2, marker="o")

# ── accuracy annotations ──────────────────────────────────────────────────────
for ax, clf in [(ax_l, clf_linear), (ax_r, clf_poly)]:
    acc = np.mean(clf.predict(pts) == labels)
    ax.text(0, -2.35,
            f"Train accuracy: {acc:.0%}",
            color=WHITE, fontsize=11, fontweight="bold",
            ha="center", va="bottom", zorder=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL,
                      edgecolor=GRAY, linewidth=1.2))

# ── legend ────────────────────────────────────────────────────────────────────
legend_elems = [
    mpatches.Patch(facecolor=RED,  label="Red  (+1, above y = x)"),
    mpatches.Patch(facecolor=BLUE, label="Blue (−1, below y = x)"),
    plt.Line2D([0], [0], color=GREEN, linewidth=2.5, label="Fitted boundary"),
    plt.Line2D([0], [0], color=GOLD,  linewidth=1.5,
               linestyle="--", label="True boundary (y = x)"),

]
fig.legend(
    handles=legend_elems, loc="lower center", ncol=5,
    framealpha=0.2, facecolor=PANEL, edgecolor=GRAY,
    labelcolor=WHITE, fontsize=9, bbox_to_anchor=(0.5, 0.005),
)

# ── save ──────────────────────────────────────────────────────────────────────
out_path = "outputs/vc_dim_model.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out_path}")
plt.close(fig)
