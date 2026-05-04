"""
lasso_kkt_conditions.py

KKT conditions on the 3-D L1 ball — an octahedron polytope.

Boundary hierarchy:
  8 Faces   — all 3 weights nonzero  → every r_i pinned to ±λ
  12 Edges  — exactly 1 weight = 0   → that r_i free in [-λ, λ]  (Case 3 × 1)
  6 Vertices — exactly 2 weights = 0 → those r_i free in [-λ, λ] (Case 3 × 2, most sparse)
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0e1117"
C_FACE   = "#2563eb"   # uniform blue for all faces
C_CORNER = "#ffdd57"   # vertices: 2 weights = 0 (most sparse)

T = 1.0  # λ

# ── Octahedron geometry ───────────────────────────────────────────────────────
verts = np.array([
    [ T,  0,  0],   # 0  +x  (w1 > 0)
    [-T,  0,  0],   # 1  -x  (w1 < 0)
    [ 0,  T,  0],   # 2  +y  (w2 > 0)
    [ 0, -T,  0],   # 3  -y  (w2 < 0)
    [ 0,  0,  T],   # 4  +z  (w3 > 0)
    [ 0,  0, -T],   # 5  -z  (w3 < 0)
])

# (vertex indices for each triangular face)
face_defs = [
    [0, 2, 4],   # (+,+,+)
    [1, 2, 4],   # (-,+,+)
    [0, 3, 4],   # (+,-,+)
    [0, 2, 5],   # (+,+,-)
    [1, 3, 4],   # (-,-,+)
    [1, 2, 5],   # (-,+,-)
    [0, 3, 5],   # (+,-,-)
    [1, 3, 5],   # (-,-,-)
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9), facecolor=BG)
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG)

# ── Faces ─────────────────────────────────────────────────────────────────────
triangles = [verts[idx] for idx in face_defs]

poly = Poly3DCollection(triangles, alpha=0.40, shade=False)
poly.set_facecolor(C_FACE)
poly.set_edgecolor("#dddddd")
poly.set_linewidth(0.5)
ax.add_collection3d(poly)

# ── Edges — drawn and labelled with segment notation (e.g. AC) ───────────────
edge_defs = [
    (0,2,"AC"),(0,3,"AD"),(0,4,"AE"),(0,5,"AF"),
    (1,2,"BC"),(1,3,"BD"),(1,4,"BE"),(1,5,"BF"),
    (2,4,"CE"),(2,5,"CF"),(3,4,"DE"),(3,5,"DF"),
]
for (i, j, name) in edge_defs:
    p1, p2 = verts[i], verts[j]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color="white", alpha=0.55, lw=1.2)

# ── Vertices ──────────────────────────────────────────────────────────────────
ax.scatter(verts[:,0], verts[:,1], verts[:,2],
           color=C_CORNER, s=80, zorder=10,
           edgecolors="white", linewidths=1.0, depthshade=False)

# ── Vertex letter labels (A–F, placed outside each corner) ───────────────────
O = 1.32   # offset factor
vertex_letters = [
    (0, "A"),   # +x
    (1, "B"),   # -x
    (2, "C"),   # +y
    (3, "D"),   # -y
    (4, "E"),   # +z
    (5, "F"),   # -z
]
for (vi, letter) in vertex_letters:
    lx, ly, lz = verts[vi] * O
    ax.text(lx, ly, lz, letter,
            color=C_CORNER, fontsize=13, fontweight="bold",
            ha="center", va="center")

# ── Axis style ────────────────────────────────────────────────────────────────
ax.set_xlabel("$w_1$", color="white", fontsize=12, labelpad=6)
ax.set_ylabel("$w_2$", color="white", fontsize=12, labelpad=6)
ax.set_zlabel("$w_3$", color="white", fontsize=12, labelpad=6)
ax.tick_params(colors="white", labelsize=8)
# 3D axes don't fully respect tick_params — set each axis explicitly
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.label.set_color("white")
    for t in axis.get_ticklabels():
        t.set_color("white")

for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#2a2a4a")
ax.grid(True)
ax.xaxis._axinfo["grid"].update(color="#2a2a4a", linewidth=0.6)
ax.yaxis._axinfo["grid"].update(color="#2a2a4a", linewidth=0.6)
ax.zaxis._axinfo["grid"].update(color="#2a2a4a", linewidth=0.6)
ax.view_init(elev=22, azim=32)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Conditions on the L1 Ball — 3D Polytope (Octahedron)\n"
    r"$r_i \equiv x_i^T(y - Xw^*)$  |  sparsity increases: Faces → Edges → Vertices",
    color="white", fontsize=12, fontweight="bold", pad=14)

# ── Legend: two side-by-side boxes ───────────────────────────────────────────
leg_kw = dict(facecolor="#161b27", edgecolor="#2a2a4a",
              labelcolor="white", fontsize=8.8)
dot = dict(marker="o", color="none", markerfacecolor=C_CORNER,
           markeredgecolor="white", markersize=7)
eln = dict(color="#aaaadd", lw=1.5)

vertex_handles = [
    mpatches.Patch(color=C_FACE, alpha=0.6,
        label=r"Face: all $w_i\!\neq\!0$, $r_i=\pm\lambda$"),
    Line2D([0],[0], **dot, label=r"A  ($w_1>0$):  $w_2=w_3=0$"),
    Line2D([0],[0], **dot, label=r"B  ($w_1<0$):  $w_2=w_3=0$"),
    Line2D([0],[0], **dot, label=r"C  ($w_2>0$):  $w_1=w_3=0$"),
    Line2D([0],[0], **dot, label=r"D  ($w_2<0$):  $w_1=w_3=0$"),
    Line2D([0],[0], **dot, label=r"E  ($w_3>0$):  $w_1=w_2=0$"),
    Line2D([0],[0], **dot, label=r"F  ($w_3<0$):  $w_1=w_2=0$"),
]
edge_handles = [
    Line2D([0],[0], **eln, label=r"AC  ($w_3=0$)"),
    Line2D([0],[0], **eln, label=r"AD  ($w_3=0$)"),
    Line2D([0],[0], **eln, label=r"AE  ($w_2=0$)"),
    Line2D([0],[0], **eln, label=r"AF  ($w_2=0$)"),
    Line2D([0],[0], **eln, label=r"BC  ($w_3=0$)"),
    Line2D([0],[0], **eln, label=r"BD  ($w_3=0$)"),
    Line2D([0],[0], **eln, label=r"BE  ($w_2=0$)"),
    Line2D([0],[0], **eln, label=r"BF  ($w_2=0$)"),
    Line2D([0],[0], **eln, label=r"CE  ($w_1=0$)"),
    Line2D([0],[0], **eln, label=r"CF  ($w_1=0$)"),
    Line2D([0],[0], **eln, label=r"DE  ($w_1=0$)"),
    Line2D([0],[0], **eln, label=r"DF  ($w_1=0$)"),
]

leg1 = fig.legend(handles=vertex_handles,
                  title="Vertices", title_fontsize=9,
                  loc="lower left", bbox_to_anchor=(0.01, 0.01),
                  ncol=1, **leg_kw)
leg1.get_title().set_color("white")
fig.add_artist(leg1)

leg2 = fig.legend(handles=edge_handles,
                  title=r"Edges  (1 weight $=0$ → that $r_i\in[-\lambda,\lambda]$)",
                  title_fontsize=9,
                  loc="lower right", bbox_to_anchor=(0.99, 0.01),
                  ncol=2, **leg_kw)
leg2.get_title().set_color("white")

plt.tight_layout()
out = OUTPUT_DIR / f"{stamp}_lasso_kkt_conditions.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
plt.close(fig)
