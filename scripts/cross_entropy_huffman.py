"""
cross_entropy_huffman.py

For each of K probability distributions P:
  1. Build the Huffman code for P → code lengths l(x) and binary codewords per symbol
  2. Derive the implied Huffman distribution  Q(x) = 2^{-l(x)}
     (For a complete Huffman code, sum_x Q(x) = 1 by Kraft equality.)
  3. Compute H(P, Q) = -sum_x P(x) log2 Q(x) = sum_x P(x) l(x)
     i.e. the expected Huffman code length under P.
  4. Render a figure with two panels:
       left  – Distribution P bar chart
       right – The Huffman tree for P, with leaf nodes showing symbol/codeword/prob
               and internal nodes showing combined probability; edges labelled 0/1
  5. Save one PNG per distribution to outputs/<run_tag>_cross_entropy_huffman_frames/
     then encode an mp4 slideshow.
"""

import heapq
import shutil
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import petname

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Aesthetics (matches huffman_pmf_video.py / cross_entropy.py)
# ---------------------------------------------------------------------------
BAR_COLOR_P = "#FFFFFF"   # white  – distribution P bars
BAR_COLOR_Q = "#E87B4C"   # orange – Huffman leaf nodes
BG_DARK     = "#1E1E1E"
BG_MID      = "#2A2A2A"
TEXT_COLOR  = "#DDDDDD"
WHITE       = "white"

ALPHABET = list("abcdefghijklmnopqrstuvwxyz")

# ---------------------------------------------------------------------------
# PMF generators
# ---------------------------------------------------------------------------

def uniform_pmf(n: int = 8) -> tuple[str, dict[str, float]]:
    symbols = ALPHABET[:n]
    p = 1.0 / n
    return "Uniform", {s: p for s in symbols}


def geometric_pmf(p: float = 0.4, n: int = 8) -> tuple[str, dict[str, float]]:
    symbols = ALPHABET[:n]
    probs = np.array([(1 - p) ** i * p for i in range(n)], dtype=float)
    probs[-1] += 1.0 - probs.sum()
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


DISTRIBUTIONS = [
    uniform_pmf(),
    geometric_pmf(),
    gaussian_pmf(),
    zipf_pmf(),
    bimodal_pmf(),
]

# ---------------------------------------------------------------------------
# Huffman tree
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("prob", "symbol", "left", "right")

    def __init__(self, prob, symbol=None, left=None, right=None):
        self.prob   = prob
        self.symbol = symbol
        self.left   = left
        self.right  = right

    def __lt__(self, other):
        return self.prob < other.prob


def build_huffman_tree(pmf: dict[str, float]) -> _Node:
    """Build and return the root of the Huffman tree for pmf."""
    active = [_Node(p, s) for s, p in pmf.items() if p > 0]
    if len(active) == 1:
        # Single symbol: wrap so the leaf has depth 1
        leaf = active[0]
        return _Node(leaf.prob, left=leaf)
    heapq.heapify(active)
    while len(active) > 1:
        lo = heapq.heappop(active)
        hi = heapq.heappop(active)
        heapq.heappush(active, _Node(lo.prob + hi.prob, left=lo, right=hi))
    return active[0]


def get_codes_and_lengths(
    root: _Node,
) -> tuple[dict[str, str], dict[str, int]]:
    """Traverse the tree and return (codewords, code_lengths) for every symbol."""
    codes: dict[str, str]   = {}
    lengths: dict[str, int] = {}

    def _walk(node: _Node, bits: str) -> None:
        if node.symbol is not None:          # leaf
            codes[node.symbol]   = bits or "0"
            lengths[node.symbol] = len(bits) if bits else 1
        else:
            if node.left:  _walk(node.left,  bits + "0")
            if node.right: _walk(node.right, bits + "1")

    _walk(root, "")
    return codes, lengths


def huffman_q_pmf(lengths: dict[str, int]) -> dict[str, float]:
    """Q(x) = 2^{-l(x)}.  Sums to 1 by Kraft equality for a complete code."""
    return {s: 2.0 ** (-l) for s, l in lengths.items()}

# ---------------------------------------------------------------------------
# Information-theoretic quantities
# ---------------------------------------------------------------------------
_EPS = 1e-12


def cross_entropy(p_pmf: dict[str, float], q_pmf: dict[str, float]) -> float:
    """H(P, Q) = -sum_x P(x) log2 Q(x)"""
    return -sum(
        p * np.log2(max(q_pmf.get(s, _EPS), _EPS))
        for s, p in p_pmf.items()
        if p > 0
    )


def entropy(p_pmf: dict[str, float]) -> float:
    """H(P) = -sum_x P(x) log2 P(x)"""
    return -sum(p * np.log2(p) for p in p_pmf.values() if p > 0)


def sample_counts(
    p_pmf: dict[str, float],
    n: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Draw n samples from P and return per-symbol counts as a dict."""
    symbols = list(p_pmf.keys())
    probs   = np.array([p_pmf[s] for s in symbols])
    idx     = rng.choice(len(symbols), size=n, p=probs)
    counts  = np.bincount(idx, minlength=len(symbols))
    return dict(zip(symbols, counts.tolist()))

# ---------------------------------------------------------------------------
# Tree layout & drawing
# ---------------------------------------------------------------------------

def _assign_positions(root: _Node) -> dict[int, tuple[float, float]]:
    """
    Leaves are placed at x = 0, 1, 2, … (left-to-right in-order).
    Internal nodes are centred over their children.
    Root is at y = 0; each level down decrements y by 1.
    """
    pos: dict[int, tuple[float, float]] = {}
    counter = [0]

    def _place(node: _Node, depth: int) -> None:
        is_leaf = node.left is None and node.right is None
        if is_leaf:
            pos[id(node)] = (float(counter[0]), float(-depth))
            counter[0] += 1
        else:
            if node.left:  _place(node.left,  depth + 1)
            if node.right: _place(node.right, depth + 1)
            child_xs = []
            if node.left:  child_xs.append(pos[id(node.left)][0])
            if node.right: child_xs.append(pos[id(node.right)][0])
            pos[id(node)] = (sum(child_xs) / len(child_xs), float(-depth))

    _place(root, 0)
    return pos


def draw_huffman_tree(
    ax,
    root: _Node,
    counts: dict[str, int],
    n_samples: int,
    q_dist_name: str = "",
) -> None:
    """Draw the Huffman tree on *ax* (axis decorations turned off)."""
    pos = _assign_positions(root)

    all_nodes: list[_Node] = []
    def _collect(node: _Node) -> None:
        all_nodes.append(node)
        if node.left:  _collect(node.left)
        if node.right: _collect(node.right)
    _collect(root)

    xs = [pos[id(n)][0] for n in all_nodes]
    ys = [pos[id(n)][1] for n in all_nodes]

    # Node radii in data coordinates (predictable once aspect="equal")
    LEAF_R  = 0.28
    INT_R   = 0.18

    # Scale bars so the tallest bar = BAR_MAX data units
    BAR_MAX  = 0.50
    max_frac = max((counts.get(n.symbol, 0) / n_samples
                    for n in all_nodes if n.symbol is not None), default=1)
    bar_scale = BAR_MAX / max_frac if max_frac > 0 else 1.0

    # y budget below each leaf: ring (LEAF_R) + text gap + prob text (~0.18)
    #                           + bar gap + BAR_MAX + fraction label (~0.20)
    BELOW_LEAF = LEAF_R + 0.12 + 0.18 + 0.12 + BAR_MAX + 0.20

    xmin, xmax = min(xs) - 0.70, max(xs) + 0.70
    ymin = min(ys) - BELOW_LEAF
    ymax = 0.55

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor(BG_MID)
    ax.axis("off")
    ax.set_title(f"Huffman Tree for Q = {q_dist_name} Sampled by Distribution P",
                 color=WHITE, fontsize=12, pad=8)

    # ── edges ────────────────────────────────────────────────────────────────
    def _draw_edges(node: _Node) -> None:
        nx, ny = pos[id(node)]
        for child, bit in [(node.left, "0"), (node.right, "1")]:
            if child is None:
                continue
            cx, cy = pos[id(child)]
            ax.plot([nx, cx], [ny, cy], color="#555555", lw=1.1, zorder=1)
            mx = nx + (cx - nx) * 0.30
            my = ny + (cy - ny) * 0.30
            ax.text(mx, my, bit,
                    color="#AAAAAA", fontsize=7, ha="center", va="center", zorder=3,
                    bbox=dict(facecolor=BG_MID, edgecolor="none", pad=0.6, alpha=0.85))
            _draw_edges(child)

    _draw_edges(root)

    # ── nodes ────────────────────────────────────────────────────────────────
    def _draw_nodes(node: _Node) -> None:
        nx, ny = pos[id(node)]
        is_leaf = node.left is None and node.right is None

        if is_leaf:
            # White ring for contrast, then orange circle
            ax.add_patch(plt.Circle((nx, ny), LEAF_R + 0.03, color=WHITE, zorder=4))
            ax.add_patch(plt.Circle((nx, ny), LEAF_R, color=BAR_COLOR_Q, zorder=5))
            ax.text(nx, ny, node.symbol,
                    color=WHITE, fontsize=9, fontweight="bold",
                    ha="center", va="center", zorder=6)

            # Orange Q(x) probability — anchored just below circle edge
            prob_y = ny - LEAF_R - 0.12
            ax.text(nx, prob_y, f"{node.prob:.1%}",
                    color=BAR_COLOR_Q, fontsize=6, fontweight="bold",
                    ha="center", va="top", zorder=6)

            # White sample bar — starts below the probability text
            bar_top    = prob_y - 0.18
            frac       = counts.get(node.symbol, 0) / n_samples
            bar_h      = frac * bar_scale
            bar_bottom = bar_top - bar_h
            ax.bar([nx], [bar_h], bottom=[bar_bottom],
                   width=0.45, color=BAR_COLOR_P, alpha=0.85,
                   zorder=3, linewidth=0)

            # White fraction label below the bar
            ax.text(nx, bar_bottom - 0.06, f"{frac:.1%}",
                    color=BAR_COLOR_P, fontsize=6, fontweight="bold",
                    ha="center", va="top", zorder=6)
        else:
            ax.add_patch(plt.Circle((nx, ny), INT_R + 0.02, color="#888888", zorder=4))
            ax.add_patch(plt.Circle((nx, ny), INT_R, color="#3C3C3C", zorder=5))
            ax.text(nx, ny, f"{node.prob:.2f}",
                    color=TEXT_COLOR, fontsize=6,
                    ha="center", va="center", zorder=6)

        if node.left:  _draw_nodes(node.left)
        if node.right: _draw_nodes(node.right)

    _draw_nodes(root)

    # ── legend — placed in the corner furthest from any node ─────────────────
    corners = {
        "upper right": (xmax, ymax),
        "upper left":  (xmin, ymax),
        "lower right": (xmax, ymin),
        "lower left":  (xmin, ymin),
    }
    def _min_node_dist(cx, cy):
        return min(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 for x, y in zip(xs, ys))
    best_loc = max(corners, key=lambda k: _min_node_dist(*corners[k]))

    ax.legend(
        handles=[
            mpatches.Patch(color=BAR_COLOR_Q, label="Orange number: Q(symbol) "),
            mpatches.Patch(color=BAR_COLOR_P, alpha=0.85, label="White bar: P(symbol)"),
        ],
        loc=best_loc, fontsize=7,
        facecolor=BG_MID, edgecolor="#444444", labelcolor=TEXT_COLOR,
    )

# ---------------------------------------------------------------------------
# Frame renderer
# ---------------------------------------------------------------------------

def make_frame(
    p_name: str,
    p_pmf: dict[str, float],
    q_name: str,
    q_pmf_source: dict[str, float],
    out_path: str,
    n_samples: int = 100_000,
    rng: np.random.Generator | None = None,
) -> None:
    if rng is None:
        rng = np.random.default_rng()

    # Q is the Huffman-implied distribution built from q_pmf_source
    root           = build_huffman_tree(q_pmf_source)
    codes, lengths = get_codes_and_lengths(root)
    q_pmf          = huffman_q_pmf(lengths)
    counts         = sample_counts(p_pmf, n_samples, rng)

    h_pq = cross_entropy(p_pmf, q_pmf)   # = sum_x P(x) * l_Q(x)

    symbols  = list(p_pmf.keys())
    p_probs  = [p_pmf[s]        for s in symbols]
    q_probs  = [q_pmf.get(s, 0) for s in symbols]

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 6), facecolor=BG_DARK)
    fig.suptitle(
        f"Cross Entropy of P and Q: H(P, Q) = {h_pq:.4f} bits",
        fontsize=15, fontweight="bold", color=WHITE, y=0.98,
    )
    fig.text(
        0.5, 0.91,
        r"$H(P,Q) = -\sum_i P(x_i)\log Q(x_i) = \sum_i P(x_i)\, I_Q(x_i)$",
        ha="center", va="center", fontsize=11, color=TEXT_COLOR,
    )

    # Three panels: P bar chart | Huffman tree | Q bar chart
    gs = fig.add_gridspec(
        1, 3,
        left=0.05, right=0.98,
        top=0.84,  bottom=0.10,
        wspace=0.12,
        width_ratios=[1, 2, 1],
    )
    ax_p    = fig.add_subplot(gs[0])
    ax_tree = fig.add_subplot(gs[1])
    ax_q    = fig.add_subplot(gs[2])

    # ── left: P bar chart ────────────────────────────────────────────────────
    for ax in (ax_p, ax_q):
        ax.set_facecolor(BG_MID)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    ax_p.set_title(f"P = Probability Mass Function for {p_name}", color=WHITE, fontsize=12, pad=8)
    ax_p.bar(symbols, p_probs,
             color=BAR_COLOR_P, edgecolor="#222222", linewidth=0.8)
    ax_p.set_xlabel("Symbol", color=TEXT_COLOR, fontsize=9)
    ax_p.set_ylabel("Probability", color=TEXT_COLOR, fontsize=9)
    ax_p.set_ylim(0, 1)
    ax_p.set_xticklabels(
        [f"{s}\n{p:.1%}" for s, p in zip(symbols, p_probs)],
        color=TEXT_COLOR, fontsize=8,
    )

    # ── centre: Huffman tree ──────────────────────────────────────────────────
    for spine in ax_tree.spines.values():
        spine.set_edgecolor("#444444")
    draw_huffman_tree(ax_tree, root, counts, n_samples, q_dist_name=q_name)

    # ── right: Q PMF bar chart ────────────────────────────────────────────────
    ax_q.set_title(f"Q = Probability Mass Function for {q_name}", color=WHITE, fontsize=12, pad=8)
    ax_q.bar(symbols, q_probs,
             color=BAR_COLOR_Q, edgecolor="#222222", linewidth=0.8)
    ax_q.set_xlabel("Symbol", color=TEXT_COLOR, fontsize=9)
    ax_q.set_ylabel("Probability", color=TEXT_COLOR, fontsize=9)
    ax_q.set_ylim(0, 1)
    ax_q.set_xticklabels(
        [f"{s}\n{q:.1%}" for s, q in zip(symbols, q_probs)],
        color=TEXT_COLOR, fontsize=8,
    )

    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stamp   = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    name    = petname.generate(2, "_")
    run_tag = f"{stamp}_{name}"

    tmp_dir = OUTPUT_DIR / f"{run_tag}_cross_entropy_huffman_frames"
    tmp_dir.mkdir(exist_ok=True)

    rng   = np.random.default_rng(42)
    pairs = list(product(range(len(DISTRIBUTIONS)), repeat=2))

    k = len(DISTRIBUTIONS)
    print(f"K = {k} distributions  →  K² = {len(pairs)} ordered pairs (including P==Q)")

    frame_paths = []
    for idx, (pi, qi) in enumerate(pairs):
        p_name, p_pmf = DISTRIBUTIONS[pi]
        q_name, q_pmf_source = DISTRIBUTIONS[qi]

        safe_p   = p_name.split()[0]
        safe_q   = q_name.split()[0]
        out_path = tmp_dir / f"frame_{idx:03d}_P={safe_p}_Q={safe_q}.png"

        print(f"  [{idx + 1:2d}/{len(pairs)}]  H({p_name} ‖ Huffman({q_name})) …")
        make_frame(p_name, p_pmf, q_name, q_pmf_source, str(out_path),
                   n_samples=100_000, rng=rng)
        frame_paths.append(out_path)

    # Build video with ffmpeg concat demuxer – 3 seconds per frame
    FPS          = 30
    DURATION_SEC = 3

    concat_file = tmp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for fp in frame_paths:
            f.write(f"file '{fp.resolve()}'\n")
            f.write(f"duration {DURATION_SEC}\n")
        f.write(f"file '{frame_paths[-1].resolve()}'\n")

    out_video = OUTPUT_DIR / f"{run_tag}_cross_entropy_huffman.mp4"
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

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
