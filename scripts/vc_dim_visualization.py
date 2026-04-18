import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from itertools import product
from tqdm import tqdm

# ── colours ──────────────────────────────────────────────────────────────────
RED   = "#E74C3C"
BLUE  = "#3498DB"
BG    = "#000000"
PANEL = "#000000"
GOLD  = "#FFFFFF"
WHITE = "#EAEAEA"
GREEN = "#2ECC71"
GRAY  = "#7F8C8D"

# ── point layouts ─────────────────────────────────────────────────────────────
# 3 points: rough triangle (general position)
PTS3 = np.array([[-1.0, -0.7], [1.0, -0.7], [0.0,  0.9]])

# 4 points: square; the XOR / opposite-corners cases break linear separation
PTS4 = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0,  1.0], [-1.0,  1.0]])

# ── helpers ───────────────────────────────────────────────────────────────────

def find_separator(pts, labels):
    """
    Find the max-margin linear separator for `labels` (+1 / -1) on `pts`.
    For each candidate normal direction w, the valid bias range is solved
    analytically, and the midpoint (max margin for that direction) is chosen.
    Returns (found, w, b) where the line is w·x + b = 0.
    """
    best = None  # (margin, w, bias)
    for angle in np.linspace(0, 2 * np.pi, 1440):
        w = np.array([np.cos(angle), np.sin(angle)])
        scores = pts @ w                     # signed projection, shape (n,)
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == -1]
        # Need: scores[i] + bias > 0 for +1 class  → bias > -min(pos_scores)
        # Need: scores[i] + bias < 0 for -1 class  → bias < -max(neg_scores)
        lo = -np.min(pos_scores) if len(pos_scores) > 0 else -3.0
        hi = -np.max(neg_scores) if len(neg_scores) > 0 else  3.0
        if lo < hi:
            margin = (hi - lo) / 2
            if best is None or margin > best[0]:
                best = (margin, w, (lo + hi) / 2)
    if best is not None:
        return True, best[1], best[2]
    return False, None, None


def draw_separator(ax, w, b, color=GREEN, alpha=0.9):
    """Draw the decision line and shade the two half-planes."""
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())

    # ── shade half-planes via contourf on a meshgrid ──────────────────────────
    xs = np.linspace(xlim[0], xlim[1], 300)
    ys = np.linspace(ylim[0], ylim[1], 300)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = w[0] * XX + w[1] * YY + b   # positive side → red (+1), negative → blue (-1)

    ax.contourf(XX, YY, ZZ, levels=[-1e6, 0, 1e6],
                colors=["#3498DB", "#E74C3C"], alpha=0.18, zorder=1)

    # ── draw the boundary line ────────────────────────────────────────────────
    x_vals = xlim.copy()
    if abs(w[1]) > 1e-6:
        y_vals = -(w[0] * x_vals + b) / w[1]
    else:
        x_vals = np.array([-b / w[0], -b / w[0]])
        y_vals = ylim.copy()
    ax.plot(x_vals, y_vals, color=color, linewidth=2.5, alpha=alpha, zorder=4)


def coloring_label(labels):
    """Return a short string like 'RBRB' for the coloring."""
    return "".join("R" if l == 1 else "B" for l in labels)


# ── build all frames ──────────────────────────────────────────────────────────

def build_frames_3pts():
    """All 8 colorings of 3 points – each is shattering-success."""
    frames = []
    colorings = list(product([1, -1], repeat=3))
    for bits in tqdm(colorings, desc="  3-pt colorings", unit="coloring"):
        labels = np.array(bits)
        found, w, b = find_separator(PTS3, labels)
        frames.append(dict(pts=PTS3, labels=labels, found=found, w=w, b=b,
                           title=f"Coloring: {coloring_label(labels)}",
                           success_msg="Separable ✓" if found else "NOT separable ✗"))
    return frames


def build_frames_4pts():
    """All 16 colorings of 4 points – highlight the failures."""
    frames = []
    colorings = list(product([1, -1], repeat=4))
    for bits in tqdm(colorings, desc="  4-pt colorings", unit="coloring"):
        labels = np.array(bits)
        found, w, b = find_separator(PTS4, labels)
        frames.append(dict(pts=PTS4, labels=labels, found=found, w=w, b=b,
                           title=f"Coloring: {coloring_label(labels)}",
                           success_msg="Separable ✓" if found else "NOT separable ✗"))
    return frames


print("Building 3-point frames...")
frames3 = build_frames_3pts()   # 8 frames
print("Building 4-point frames...")
frames4 = build_frames_4pts()   # 16 frames

# Cumulative separable counts as we progress through each unique sequence
def cumulative_counts(frames):
    counts, c = [], 0
    for f in frames:
        if f["found"]: c += 1
        counts.append(c)
    return counts

counts3 = cumulative_counts(frames3)   # length 8
counts4 = cumulative_counts(frames4)   # length 16

# Pair them up: repeat 3-pt frames so both sides advance together (lcm = 16)
import math
n3, n4 = len(frames3), len(frames4)
total = math.lcm(n3, n4)          # 16
seq3  = [(frames3[i % n3], counts3[i % n3]) for i in range(total)]
seq4  = [(frames4[i % n4], counts4[i % n4]) for i in range(total)]

# ── figure / animation ────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 7), facecolor=BG)
fig.patch.set_facecolor(BG)

# Super-title
fig.text(0.5, 0.96, "VC Dimension — Linear Model = 3",
         ha="center", va="top", fontsize=20, fontweight="bold",
         color=GOLD, fontfamily="monospace")

# Left panel (3 pts)
ax_l = fig.add_axes([0.04, 0.08, 0.44, 0.80], facecolor=PANEL)
# Right panel (4 pts)
ax_r = fig.add_axes([0.52, 0.08, 0.44, 0.80], facecolor=PANEL)

for ax, label in [(ax_l, "3 Points — VC dim = 3\n(all 8 colorings separable)"),
                  (ax_r, "4 Points — VC dim > 3?\n(some colorings NOT separable)")]:
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.set_title(label, color=WHITE, fontsize=11, pad=8)

# Static legend
legend_elems = [
    mpatches.Patch(facecolor=RED,   label="Red (+1)"),
    mpatches.Patch(facecolor=BLUE,  label="Blue (−1)"),
    plt.Line2D([0], [0], color=GREEN, linewidth=2, label="Decision boundary"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=3,
           framealpha=0.2, facecolor=PANEL, edgecolor=GRAY,
           labelcolor=WHITE, fontsize=9, bbox_to_anchor=(0.5, 0.01))


def draw_panel(ax, frame, panel_idx, count, total_colorings):
    ax.cla()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)

    pts    = frame["pts"]
    labels = frame["labels"]
    found  = frame["found"]
    w      = frame["w"]
    b      = frame["b"]

    # Draw points
    for i, (p, lbl) in enumerate(zip(pts, labels)):
        c = RED if lbl == 1 else BLUE
        ax.scatter(*p, s=220, color=c, zorder=5, edgecolors=WHITE, linewidths=1.2)
        ax.text(p[0], p[1] + 0.18, f"p{i+1}", color=WHITE, fontsize=8,
                ha="center", va="bottom", zorder=6)

    # Draw separator if found
    if found:
        draw_separator(ax, w, b)

    # Coloring label
    ax.set_title(f"{frame['title']}", color=WHITE, fontsize=11, pad=6)

    # Success / failure badge
    msg   = frame["success_msg"]
    color = GREEN if found else RED
    ax.text(0, -1.45, msg, color=color, fontsize=13, fontweight="bold",
            ha="center", va="bottom", zorder=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL,
                      edgecolor=color, linewidth=1.5))

    # Panel sub-header (upper-left)
    if panel_idx == 0:
        ax.text(-1.85, 1.45, "3-pt shatter", color=GOLD, fontsize=8, va="top")
    else:
        ax.text(-1.85, 1.45, "4-pt attempt", color=GOLD, fontsize=8, va="top")

    # Separable counter (upper-right)
    ax.text(1.85, 1.45, f"{count}/{total_colorings} separable",
            color=WHITE, fontsize=10, fontweight="bold",
            ha="right", va="top", zorder=7)


HOLD_FRAMES = 45   # frames each coloring is displayed (~1.5 s at 30 fps)

def make_frame_data():
    """Expand each logical frame into HOLD_FRAMES identical animation frames."""
    data = []
    for (f3, c3), (f4, c4) in zip(seq3, seq4):
        for _ in range(HOLD_FRAMES):
            data.append((f3, c3, f4, c4))
    return data

frame_data = make_frame_data()


def animate(i):
    f3, c3, f4, c4 = frame_data[i]
    draw_panel(ax_l, f3, 0, c3, n3)
    draw_panel(ax_r, f4, 1, c4, n4)
    return ax_l, ax_r


anim = FuncAnimation(fig, animate, frames=len(frame_data),
                     interval=1000 / 30, blit=False)

# ── save ──────────────────────────────────────────────────────────────────────
out_path = "outputs/vc_dim_visualization.mp4"
writer   = FFMpegWriter(fps=30, metadata=dict(title="VC Dimension"), bitrate=2000)

total_frames = len(frame_data)
with tqdm(total=total_frames, desc="Rendering", unit="frame") as pbar:
    anim.save(out_path, writer=writer,
              progress_callback=lambda i, n: pbar.update(1))

print(f"Saved → {out_path}")
plt.close(fig)
