import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

# ── Colors (matching project palette) ────────────────────────────────────────
PURPLE = "#FF10F0"
BG     = "#0D0D0D"
GOLD   = "#F0C040"
WHITE  = "#EAEAEA"
GRAY   = "#AAAAAA"
FADED  = "#FFFFFF"   # fill for out-of-subset points
FADED_EC = "#FFFFFF" # edge for out-of-subset points

# ── Point layout ─────────────────────────────────────────────────────────────
# 5 old points + 1 new point x_N  →  N = 6, k = 3
old_pts = np.array([
    [-1.8,  0.3],  # 0  left cluster
    [-1.5, -0.9],  # 1  left-bottom
    [-0.5,  0.9],  # 2  right of cluster
    [-0.3, -0.5],  # 3  right of cluster, closer to xN
    [-1.1,  1.1],  # 4  top
])
xN = np.array([1.5, 0.2])

# One color for all in-subset points
labels = [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE]

# Case 1:  k-subset is entirely among old points  →  Contributes B(N-1, k)
C1_old_idx = [0, 4, 2]          # 3 old points inside the oval

# Case 2:  k-subset contains x_N  →  remaining k-1 points from old  →  B(N-1, k-1)
C2_old_idx = [2, 3]             # k-1 = 2 old points inside the oval with x_N


# ── Helper: bounding ellipse with padding ────────────────────────────────────
def bounding_ellipse(pts, pad_x=0.55, pad_y=0.50):
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    dx = pts[:, 0].max() - pts[:, 0].min()
    dy = pts[:, 1].max() - pts[:, 1].min()
    w = max(dx + 2 * pad_x, 1.1)
    h = max(dy + 2 * pad_y, 0.9)
    return cx, cy, w, h


# ── Draw one panel ────────────────────────────────────────────────────────────
def draw_panel(ax, case):
    ax.set_facecolor(BG)
    ax.set_aspect('equal')
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.7, 1.7)
    ax.axis('off')

    if case == 1:
        old_subset   = set(C1_old_idx)
        xN_in_subset = False
        oval_pts     = old_pts[C1_old_idx]
        membership   = r'$x_N \notin S$'
        contrib      = r'Contributes  $B(N\!-\!1,\; k)$'
    else:
        old_subset   = set(C2_old_idx)
        xN_in_subset = True
        oval_pts     = np.vstack([old_pts[C2_old_idx], xN])
        membership   = r'$x_N \in S$'
        contrib      = r'Contributes  $B(N\!-\!1,\; k\!-\!1)$'

    # ── Dashed gold oval ──────────────────────────────────────────────────────
    cx, cy, w, h = bounding_ellipse(oval_pts)
    ellipse = Ellipse(
        (cx, cy), width=w, height=h,
        fill=False, edgecolor=GOLD, linestyle='--', linewidth=2.2, zorder=2
    )
    ax.add_patch(ellipse)

    # ── Old points ────────────────────────────────────────────────────────────
    for i, (pt, col) in enumerate(zip(old_pts, labels)):
        in_sub = (i in old_subset)
        ax.scatter(
            pt[0], pt[1],
            s       = 200 if in_sub else 130,
            c       = col if in_sub else FADED,
            edgecolors = WHITE if in_sub else FADED_EC,
            linewidths = 1.8 if in_sub else 1.0,
            alpha   = 1.0,
            zorder  = 4 if in_sub else 3,
        )

    # ── x_N star ─────────────────────────────────────────────────────────────
    ax.scatter(
        xN[0], xN[1],
        s          = 420 if xN_in_subset else 280,
        c          = GOLD,
        edgecolors = WHITE if xN_in_subset else FADED_EC,
        linewidths = 2.2,
        marker     = '*',
        alpha      = 1.0,
        zorder     = 5,
    )



# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2, figsize=(13, 6.2), facecolor=BG,
    gridspec_kw={'wspace': 0.08}
)
fig.subplots_adjust(left=0.03, right=0.97, top=0.78, bottom=0.14)

draw_panel(axes[0], case=1)
draw_panel(axes[1], case=2)

# ── Panel labels ──────────────────────────────────────────────────────────────
axes[0].set_title(
    'Case 1\n' + r'$x_N \notin S$  —  Contributes $B(N\!-\!1,\; k)$',
    color=WHITE, fontsize=13, pad=8, linespacing=1.6
)
axes[1].set_title(
    'Case 2\n' + r'$x_N \in S$  —  Contributes $B(N\!-\!1,\; k\!-\!1)$',
    color=WHITE, fontsize=13, pad=8, linespacing=1.6
)

# ── Main title ────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.97,
    r'$B(N,k)\;=\;B(N\!-\!1,\,k)\;+\;B(N\!-\!1,\,k\!-\!1)$',
    ha='center', va='top', color=WHITE, fontsize=18, fontweight='bold'
)

# ── Subtitle ──────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.90,
    'Every size-$k$ subset of $N$ points either contains $x_N$ or it does not.',
    ha='center', va='top', color=WHITE, fontsize=11
)

# ── Thin divider between panels ───────────────────────────────────────────────
fig.add_artist(
    plt.Line2D([0.5, 0.5], [0.133, 0.863],
               transform=fig.transFigure,
               color='white', linewidth=2.0, alpha=1.0, zorder=0)
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=PURPLE, edgecolor=WHITE, label='In subset $S$'),
    mpatches.Patch(facecolor=WHITE, edgecolor=WHITE, label='Not in subset $S$'),
    Line2D([0], [0], marker='*', color='none', markerfacecolor=GOLD,
           markeredgecolor=WHITE, markersize=13, label=r'$x_N$  (New point)'),
    mpatches.Patch(facecolor='none', edgecolor=GOLD,
                   linestyle='--', linewidth=2, label='Chosen $k$-subset  $S$'),
]
fig.legend(
    handles=legend_handles,
    loc='lower center', ncol=5,
    framealpha=0.12, facecolor=BG, edgecolor=GRAY,
    labelcolor=WHITE, fontsize=9.5,
    bbox_to_anchor=(0.5, 0.005)
)

# ── Save ─────────────────────────────────────────────────────────────────────
out = 'outputs/vc_dim_inductive_case.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {out}")
