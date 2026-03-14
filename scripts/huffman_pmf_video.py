"""
huffman_pmf_video.py

For several probability distributions:
  1. Generate a PMF over a discrete alphabet.
  2. Build a Huffman tree from that PMF.
  3. Render a figure with three panels:
       left  – probability table
       centre – Huffman tree diagram
       right  – PMF bar chart
     with a top title showing the distribution name and Shannon entropy.
  4. Collect all frames into an MP4 (one second per distribution) saved to
     outputs/.
"""

import os
import heapq
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import petname

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Huffman tree
# ---------------------------------------------------------------------------

class HuffNode:
    """Node in a Huffman tree."""
    __slots__ = ("prob", "symbol", "left", "right", "code")

    def __init__(self, prob: float, symbol=None, left=None, right=None):
        self.prob = prob
        self.symbol = symbol  # leaf symbol, or None for internal node
        self.left = left
        self.right = right
        self.code: str = ""

    # heapq needs comparison; break ties arbitrarily
    def __lt__(self, other):
        return self.prob < other.prob


def build_huffman_tree(pmf: dict[str, float]) -> HuffNode:
    """Build and return the root of a Huffman tree."""
    heap = [HuffNode(p, s) for s, p in pmf.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        merged = HuffNode(lo.prob + hi.prob, left=lo, right=hi)
        heapq.heappush(heap, merged)
    return heap[0]


def assign_codes(node: HuffNode, prefix: str = "") -> dict[str, str]:
    """Walk the tree and return {symbol: codeword}."""
    if node is None:
        return {}
    if node.symbol is not None:          # leaf
        node.code = prefix or "0"        # single-symbol edge case
        return {node.symbol: node.code}
    codes = {}
    codes.update(assign_codes(node.left,  prefix + "0"))
    codes.update(assign_codes(node.right, prefix + "1"))
    return codes

# ---------------------------------------------------------------------------
# Tree layout (Reingold-Tilford-like via simple recursive placement)
# ---------------------------------------------------------------------------

def _tree_size(node) -> int:
    if node is None:
        return 0
    return 1 + _tree_size(node.left) + _tree_size(node.right)


def layout_tree(node, depth=0, counter=None):
    """
    Assign (x, y) positions to every node.
    Returns a dict {id(node): (x, y, node)}.
    Uses an in-order traversal counter so leaves don't overlap.
    """
    if counter is None:
        counter = [0]
    if node is None:
        return {}
    positions = {}
    positions.update(layout_tree(node.left,  depth + 1, counter))
    x = counter[0]
    y = -depth
    counter[0] += 1
    positions[id(node)] = (x, y, node)
    positions.update(layout_tree(node.right, depth + 1, counter))
    return positions

# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def entropy(pmf: dict[str, float]) -> float:
    h = 0.0
    for p in pmf.values():
        if p > 0:
            h -= p * np.log2(p)
    return h

# ---------------------------------------------------------------------------
# PMF generators
# ---------------------------------------------------------------------------

ALPHABET = list("abcdefghijklmnopqrstuvwxyz")


def uniform_pmf(n: int = 8) -> tuple[str, dict[str, float]]:
    symbols = ALPHABET[:n]
    p = 1.0 / n
    return "Uniform", {s: p for s in symbols}


def geometric_pmf(p: float = 0.4, n: int = 8) -> tuple[str, dict[str, float]]:
    symbols = ALPHABET[:n]
    probs = np.array([(1 - p) ** i * p for i in range(n)], dtype=float)
    probs[-1] += 1.0 - probs.sum()          # absorb tail
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()
    return "Geometric (p=0.4)", dict(zip(symbols, probs.tolist()))


def gaussian_pmf(n: int = 8) -> tuple[str, dict[str, float]]:
    symbols = ALPHABET[:n]
    xs = np.linspace(-2, 2, n)
    probs = np.exp(-0.5 * xs ** 2)
    probs /= probs.sum()
    return "Gaussian", dict(zip(symbols, probs.tolist()))


def zipf_pmf(n: int = 8, alpha: float = 1.5) -> tuple[str, dict[str, float]]:
    symbols = ALPHABET[:n]
    probs = np.array([1.0 / (i ** alpha) for i in range(1, n + 1)])
    probs /= probs.sum()
    return f"Zipf (α={alpha})", dict(zip(symbols, probs.tolist()))


def bimodal_pmf(n: int = 8) -> tuple[str, dict[str, float]]:
    symbols = ALPHABET[:n]
    xs = np.linspace(-3, 3, n)
    probs = np.exp(-0.5 * (xs - 1.5) ** 2) + np.exp(-0.5 * (xs + 1.5) ** 2)
    probs /= probs.sum()
    return "Bimodal", dict(zip(symbols, probs.tolist()))


def skewed_pmf(n: int = 8) -> tuple[str, dict[str, float]]:
    """Exponentially skewed – one symbol dominates."""
    symbols = ALPHABET[:n]
    probs = np.array([0.5, 0.25, 0.12, 0.06, 0.03, 0.02, 0.01, 0.01])
    probs = probs[:n]
    probs /= probs.sum()
    return "Skewed", dict(zip(symbols, probs.tolist()))


DISTRIBUTIONS = [
    uniform_pmf(),
    geometric_pmf(),
    gaussian_pmf(),
    zipf_pmf(),
    bimodal_pmf(),
    skewed_pmf(),
]

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
NODE_RADIUS = 0.52
LEAF_COLOR  = "#4C9BE8"
INT_COLOR   = "#E87B4C"
EDGE_COLOR  = "#888899"
TEXT_COLOR  = "#FFFFFF"


def draw_tree(ax, positions: dict):
    """Draw the Huffman tree on *ax* using pre-computed positions."""
    # Edges first (no 0/1 labels)
    for nid, (x, y, node) in positions.items():
        for child in (node.left, node.right):
            if child is None:
                continue
            cx, cy, _ = positions[id(child)]
            ax.plot([x, cx], [y, cy], color=EDGE_COLOR, lw=2.5, zorder=1)

    # Nodes on top
    for nid, (x, y, node) in positions.items():
        color = LEAF_COLOR if node.symbol else INT_COLOR
        # Outer ring for contrast
        ring = plt.Circle((x, y), NODE_RADIUS + 0.02, color="white", zorder=2)
        ax.add_patch(ring)
        circle = plt.Circle((x, y), NODE_RADIUS, color=color, zorder=3)
        ax.add_patch(circle)
        if node.symbol:
            label = f"{node.symbol}\n{node.prob:.3f}"
        else:
            label = f"{node.prob:.3f}"
        ax.text(x, y, label, fontsize=9, ha="center", va="center",
                color=TEXT_COLOR, zorder=4, fontweight="bold",
                linespacing=1.3)


def make_frame(dist_name: str, pmf: dict[str, float], out_path: str):
    """Render a single figure and save it to *out_path*."""
    root = build_huffman_tree(pmf)
    codes = assign_codes(root)
    h = entropy(pmf)

    positions = layout_tree(root)
    xs = [v[0] for v in positions.values()]
    ys = [v[1] for v in positions.values()]

    symbols = list(pmf.keys())
    probs   = list(pmf.values())

    # ---- figure layout ----
    fig = plt.figure(figsize=(16, 6), facecolor="#1A1A2E")
    fig.suptitle(
        f"{dist_name}      H(X) = {h:.4f} bits",
        fontsize=16, fontweight="bold", color="white", y=0.97
    )

    gs = fig.add_gridspec(1, 3, left=0.03, right=0.97, top=0.88, bottom=0.05,
                          wspace=0.18, width_ratios=[0.8, 2.4, 1.6])
    ax_table = fig.add_subplot(gs[0])
    ax_tree  = fig.add_subplot(gs[1])
    ax_bar   = fig.add_subplot(gs[2])

    for ax in (ax_table, ax_tree, ax_bar):
        ax.set_facecolor("#16213E")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # ---- LEFT: probability table ----
    ax_table.axis("off")
    ax_table.set_title("Probability Table", color="white", fontsize=11, pad=6)

    col_labels = ["Symbol", "P(x)", "Len"]
    table_data = [
        [s, f"{pmf[s]:.4f}", str(len(codes.get(s, "")))]
        for s in symbols
    ]

    # Explicit column widths: Symbol and Len are short, P(x) is wider
    col_widths = [0.18, 0.30, 0.14]
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Style table cells
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444466")
        if row == 0:
            cell.set_facecolor("#2A2A5A")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#1E2A4A" if row % 2 == 0 else "#16213E")
            cell.set_text_props(color="#CCDDFF")

    # ---- CENTRE: Huffman tree ----
    ax_tree.set_title("Huffman Tree", color="white", fontsize=11, pad=6)
    draw_tree(ax_tree, positions)

    x_pad = 1.0
    y_pad = 0.8
    ax_tree.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax_tree.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax_tree.set_aspect("equal")
    ax_tree.axis("off")

    # Legend
    leaf_patch = mpatches.Patch(color=LEAF_COLOR, label="Leaf (symbol)")
    int_patch  = mpatches.Patch(color=INT_COLOR,  label="Internal")
    ax_tree.legend(handles=[leaf_patch, int_patch], loc="upper right",
                   fontsize=7, facecolor="#16213E", edgecolor="#444466",
                   labelcolor="white")

    # ---- RIGHT: PMF bar chart ----
    ax_bar.set_title("PMF", color="white", fontsize=11, pad=6)
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(symbols)))
    bars = ax_bar.bar(symbols, probs, color=colors, edgecolor="#333355", linewidth=0.8)
    ax_bar.set_xlabel("Symbol", color="#CCDDFF", fontsize=9)
    ax_bar.set_ylabel("Probability", color="#CCDDFF", fontsize=9)
    ax_bar.tick_params(colors="#CCDDFF", labelsize=8)
    ax_bar.set_ylim(0, max(probs) * 1.2)

    # Probability value on top of each bar
    for bar, p in zip(bars, probs):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(probs) * 0.02,
            f"{p:.3f}", ha="center", va="bottom", fontsize=7, color="#CCDDFF"
        )

    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    name  = petname.generate(2, "_")
    run_tag = f"{stamp}_{name}"

    tmp_dir = OUTPUT_DIR / f"{run_tag}_huffman_frames"
    tmp_dir.mkdir(exist_ok=True)

    frame_paths = []
    for i, (dist_name, pmf) in enumerate(DISTRIBUTIONS):
        frame_file = tmp_dir / f"frame_{i:03d}.png"
        print(f"  Rendering: {dist_name} …")
        make_frame(dist_name, pmf, str(frame_file))
        frame_paths.append(frame_file)

    # Build video: each frame shown for 1 second → 1 fps with -r 1 doesn't
    # always work well; use 30 fps and duplicate frames instead.
    FPS = 30
    DURATION_SEC = 1          # seconds per distribution
    HOLD_FRAMES  = FPS * DURATION_SEC

    # Write a concat list
    concat_file = tmp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for fp in frame_paths:
            f.write(f"file '{fp.resolve()}'\n")
            f.write(f"duration {DURATION_SEC}\n")
        # ffmpeg concat demuxer needs the last file listed again (no duration)
        f.write(f"file '{frame_paths[-1].resolve()}'\n")

    out_video = OUTPUT_DIR / f"{run_tag}_huffman.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-vf", f"fps={FPS},scale=1920:-2:flags=lanczos",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(out_video),
    ]
    print("\nEncoding video …")
    subprocess.run(cmd, check=True)
    print(f"\nDone! Video saved to:\n  {out_video}")

    # Clean up frames
    import shutil
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
