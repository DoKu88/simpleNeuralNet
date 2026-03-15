"""
cross_entropy.py

For K probability distributions, generate all K*(K-1) ordered (P, Q) pairs and:
  1. Compute theoretical cross entropy H(P, Q) = -sum_x P(x) log2 Q(x)
  2. Sample n points from P and count how often each symbol is drawn
  3. Render a figure with two panels:
       left  – Distribution P bar chart
       right – Distribution Q bar chart with blue dots showing where P-samples landed
     with a top title showing H(P, Q)
  4. Save one PNG per (P, Q) pair to outputs/<run_tag>_cross_entropy/
"""

import shutil
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib.lines as mlines
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
# Aesthetics (matches huffman_pmf_video.py)
# ---------------------------------------------------------------------------
BAR_COLOR_P = "#FFFFFF"   # white  – distribution P
BAR_COLOR_Q = "#E87B4C"   # orange – distribution Q
DOT_COLOR   = "#4C9BE8"   # blue dots on Q panel
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


def sample_counts(
    p_pmf: dict[str, float],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n samples from P and return per-symbol counts."""
    symbols = list(p_pmf.keys())
    p_probs = np.array([p_pmf[s] for s in symbols])
    idx     = rng.choice(len(symbols), size=n, p=p_probs)
    return np.bincount(idx, minlength=len(symbols))

# ---------------------------------------------------------------------------
# Frame renderer
# ---------------------------------------------------------------------------

def make_frame(
    p_name: str, p_pmf: dict[str, float],
    q_name: str, q_pmf: dict[str, float],
    out_path: str,
    n_samples: int = 100_000,
    samples_per_dot: int = 1_000,
    rng: np.random.Generator | None = None,
) -> None:
    if rng is None:
        rng = np.random.default_rng()

    h_pq   = cross_entropy(p_pmf, q_pmf)
    counts = sample_counts(p_pmf, n_samples, rng)

    symbols = list(p_pmf.keys())
    p_probs = [p_pmf[s] for s in symbols]
    q_probs = [q_pmf.get(s, 0.0) for s in symbols]

    # ---- figure ----
    fig = plt.figure(figsize=(14, 6), facecolor=BG_DARK)
    fig.suptitle(
        f"Cross Entropy of P and Q: H(P, Q) = {h_pq:.4f} bits",
        fontsize=15, fontweight="bold", color=WHITE, y=0.98,
    )
    fig.text(
        0.5, 0.91,
        r"$H(P,Q) = -\sum_i P(x_i)\log Q(x_i) = \sum_i P(x_i)\, I_Q(x_i)$",
        ha="center", va="center", fontsize=11, color=TEXT_COLOR,
    )

    gs = fig.add_gridspec(
        1, 2,
        left=0.07, right=0.97,
        top=0.84,  bottom=0.13,
        wspace=0.12,
    )
    ax_p = fig.add_subplot(gs[0])
    ax_q = fig.add_subplot(gs[1])

    for ax in (ax_p, ax_q):
        ax.set_facecolor(BG_MID)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # ---- helper: annotated bar chart ----
    def bar_panel(ax, title, syms, probs, color):
        ax.set_title(title, color=WHITE, fontsize=12, pad=8)
        bars = ax.bar(syms, probs, color=color, edgecolor="#222222", linewidth=0.8)
        ax.set_xlabel("Symbol", color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel("Probability", color=TEXT_COLOR, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xticklabels(
            [f"{s}\n{p:.3f}" for s, p in zip(syms, probs)],
            color=TEXT_COLOR, fontsize=8,
        )

    bar_panel(ax_p, f"P = {p_name}", symbols, p_probs, BAR_COLOR_P)
    bar_panel(ax_q, f"Q = {q_name}", symbols, q_probs, BAR_COLOR_Q)

    # Dot matrix overlay: 5 sub-columns per bar, dots stacked bottom-to-top.
    # 1 dot = samples_per_dot samples. Max possible dots = n_samples / samples_per_dot.
    N_COLS        = 5
    MAX_DOTS_TOTAL = n_samples // samples_per_dot          # 100 for 100k / 1k
    N_MAX_ROWS    = MAX_DOTS_TOTAL // N_COLS               # 20

    bar_width   = 0.75
    col_offsets = np.linspace(-bar_width * 0.32, bar_width * 0.32, N_COLS)

    ax_dots = ax_q.twinx()
    ax_dots.set_ylim(0, 1)
    ax_dots.axis("off")

    for i, cnt in enumerate(counts):
        n_shown = cnt // samples_per_dot
        if n_shown == 0:
            continue
        dot_idx = np.arange(n_shown)
        cols    = dot_idx % N_COLS
        rows    = dot_idx // N_COLS
        xs      = i + col_offsets[cols]
        ys      = (rows + 0.5) / N_MAX_ROWS
        ax_dots.scatter(xs, ys, s=12, color=BAR_COLOR_P, zorder=6, linewidths=0)

    dot_handle = mlines.Line2D(
        [], [], marker="o", color="w", markerfacecolor=BAR_COLOR_P,
        markersize=6, label=f"1 dot = {samples_per_dot:,} samples drawn from P (n={n_samples:,})",
        linestyle="None",
    )
    ax_q.legend(handles=[dot_handle], fontsize=8, facecolor=BG_MID,
                edgecolor="#444444", labelcolor=TEXT_COLOR, loc="upper right")

    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stamp   = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    name    = petname.generate(2, "_")
    run_tag = f"{stamp}_{name}"

    tmp_dir = OUTPUT_DIR / f"{run_tag}_cross_entropy_frames"
    tmp_dir.mkdir(exist_ok=True)

    rng   = np.random.default_rng(42)
    pairs = list(product(range(len(DISTRIBUTIONS)), repeat=2))

    k = len(DISTRIBUTIONS)
    print(f"K = {k} distributions  →  K² = {len(pairs)} ordered pairs (including P==Q)")

    frame_paths = []
    for idx, (pi, qi) in enumerate(pairs):
        p_name, p_pmf = DISTRIBUTIONS[pi]
        q_name, q_pmf = DISTRIBUTIONS[qi]

        safe_p = p_name.split()[0]
        safe_q = q_name.split()[0]
        out_path = tmp_dir / f"frame_{idx:03d}_P={safe_p}_Q={safe_q}.png"

        print(f"  [{idx + 1:2d}/{len(pairs)}]  H({p_name} ‖ {q_name}) …")
        make_frame(p_name, p_pmf, q_name, q_pmf, str(out_path), rng=rng)
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

    out_video = OUTPUT_DIR / f"{run_tag}_cross_entropy.mp4"
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
