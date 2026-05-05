"""
norms_thumbnail.py

Generates a single looping thumbnail GIF that briefly rotates through each
normalization type. The norm name is displayed as a large title at the top.
No interactive widgets — output only.

Usage:
    python norms_thumbnail.py
    python norms_thumbnail.py --degrees-per-norm 120 --num-vectors 6
    python norms_thumbnail.py --output my_thumbnail.gif
"""

import argparse
import datetime
import itertools
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as mplanimation
from matplotlib import cm as mpl_cm
from tqdm import tqdm

# ====================== CLI ======================
parser = argparse.ArgumentParser(description="Thumbnail GIF cycling through all norm types")
parser.add_argument("--num-vectors", type=int, default=5, metavar="N",
                    help="Number of random vectors (default: 5)")
parser.add_argument("--gif-step", type=int, default=3, metavar="N",
                    help="Save every Nth frame (default: 3)")
parser.add_argument("--degrees-per-norm", type=float, default=90.0, metavar="DEG",
                    help="Degrees of rotation shown per norm (default: 90)")
parser.add_argument("--output", type=str, default=None, metavar="PATH",
                    help="Output GIF path (default: outputs/<timestamp>_norms_thumbnail.gif)")
args = parser.parse_args()

# ====================== CONFIG ======================
np.random.seed(42)
NUM_VECTORS = args.num_vectors
DIM = 3
AXIS_LIM = 2.0
NORMS = ["LayerNorm", "BatchNorm", "RMSNorm", "L1", "L2", "DyT"]

AZIM_STEP_DEG = 1.2        # degrees per raw frame
INTERVAL_MS = 35           # ms between raw frames
START_ELEV = 28
START_AZIM = -50

SAPPHIRE = "#0f4c99"
BASE_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]

# Title text for each norm
NORM_TITLES = {
    "LayerNorm":  "Layer Norm",
    "BatchNorm":  "Batch Norm",
    "RMSNorm":    "RMS Norm",
    "L1":         "L1 Norm",
    "L2":         "L2 Norm",
    "DyT":        "DyT (Dynamic Tanh)",
}

# Short description shown as subtitle
NORM_SUBTITLES = {
    "LayerNorm":  r"Vectors forced onto surface:  $\mu=0,\ \sigma=1$ per token  $\Rightarrow$ circle in $\mathbb{R}^3$",
    "BatchNorm":  r"Cube statistical mean:    $\mu=0,\ \sigma=1$ per feature across batch (by CLT)",
    "RMSNorm":    r"Vectors forced onto sphere surface:    $\|\mathbf{v}\|_2 = \sqrt{d}$  (no mean subtraction)",
    "L1":         r"Vectors forced onto L1 Ball surface:    $\sum_i |v_i| = 1$  $\Rightarrow$ octahedron",
    "L2":         r"Vectors forced onto sphere surface:    $\|\mathbf{v}\|_2 = 1$  $\Rightarrow$ unit sphere",
    "DyT":        r"Vectors forced inside hypercube:    $y_i = \tanh(x_i)$  $\Rightarrow$ strictly inside $(-1,1)^3$",
}


def palette(n):
    if n <= len(BASE_COLORS):
        return BASE_COLORS[:n]
    out = list(BASE_COLORS)
    cmap = mpl_cm.get_cmap("tab10")
    for i in range(len(BASE_COLORS), n):
        out.append(cmap((i - len(BASE_COLORS)) % 10))
    return out


COLORS = palette(NUM_VECTORS)

# ====================== NORMALIZATION ======================
def apply_norm(X, norm_type, eps=1e-5):
    if norm_type == "LayerNorm":
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        return (X - mean) / (std + eps)
    if norm_type == "BatchNorm":
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        return (X - mean) / (std + eps)
    if norm_type == "DyT":
        return np.tanh(X)
    if norm_type == "RMSNorm":
        rms = np.sqrt(np.mean(X ** 2, axis=1, keepdims=True) + eps)
        return X / rms
    if norm_type == "L1":
        d = np.sum(np.abs(X), axis=1, keepdims=True) + eps
        return X / d
    if norm_type == "L2":
        d = np.linalg.norm(X, axis=1, keepdims=True, ord=2) + eps
        return X / d
    return X


# ====================== FIGURE SETUP ======================
fig = plt.figure(figsize=(6, 6.8), facecolor="#ffffff")
fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.78)

ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#ffffff")
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#cccccc")
ax.tick_params(colors="#888888", labelsize=7)
ax.xaxis.label.set_color("#888888")
ax.yaxis.label.set_color("#888888")
ax.zaxis.label.set_color("#888888")
ax.set_xlabel("X", fontsize=7)
ax.set_ylabel("Y", fontsize=7)
ax.set_zlabel("Z", fontsize=7)
ax.set_xlim(-AXIS_LIM, AXIS_LIM)
ax.set_ylim(-AXIS_LIM, AXIS_LIM)
ax.set_zlim(-AXIS_LIM, AXIS_LIM)

# Title text objects (positioned in figure space)
title_text = fig.text(
    0.5, 0.93,
    "",
    ha="center", va="top",
    fontsize=22, fontweight="bold",
    color="#111111",
    fontfamily="monospace",
)
subtitle_text = fig.text(
    0.5, 0.865,
    "",
    ha="center", va="top",
    fontsize=10,
    color="#444444",
)

# ====================== CONSTRAINT SHAPES ======================
constraint_artists = []


def _clear_constraints():
    for a in constraint_artists:
        try:
            a.remove()
        except (AttributeError, ValueError):
            pass
    constraint_artists.clear()


def _sphere(radius, color, alpha=0.35, lw=1.2):
    phi = np.linspace(0, 2 * np.pi, 40)
    for t in np.linspace(0.12, np.pi - 0.12, 8):
        x = radius * np.sin(t) * np.cos(phi)
        y = radius * np.sin(t) * np.sin(phi)
        z = radius * np.cos(t) * np.ones_like(phi)
        (ln,) = ax.plot(x, y, z, color=color, alpha=alpha, linewidth=lw)
        constraint_artists.append(ln)
    thetas = np.linspace(0, np.pi, 24)
    for p in phi[::5]:
        x = radius * np.sin(thetas) * np.cos(p)
        y = radius * np.sin(thetas) * np.sin(p)
        z = radius * np.cos(thetas)
        (ln,) = ax.plot(x, y, z, color=color, alpha=alpha, linewidth=lw)
        constraint_artists.append(ln)


def _octahedron(color, alpha=0.35, lw=1.2):
    V = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
    for i in range(6):
        for j in range(i + 1, 6):
            if np.sum(np.abs(V[i] - V[j])) != 2.0:
                continue
            if np.dot(V[i], V[j]) != 0.0:
                continue
            (ln,) = ax.plot([V[i,0], V[j,0]], [V[i,1], V[j,1]], [V[i,2], V[j,2]],
                            color=color, alpha=alpha, linewidth=lw)
            constraint_artists.append(ln)


def _cube(half, color, alpha=0.35, lw=1.2):
    V = np.array(list(itertools.product([-half, half], repeat=3)), dtype=float)
    for i in range(8):
        for j in range(i + 1, 8):
            diff = V[i] - V[j]
            if np.count_nonzero(np.abs(diff) > 1e-9) != 1:
                continue
            (ln,) = ax.plot([V[i,0], V[j,0]], [V[i,1], V[j,1]], [V[i,2], V[j,2]],
                            color=color, alpha=alpha, linewidth=lw)
            constraint_artists.append(ln)


def draw_constraint_shape(norm_type):
    _clear_constraints()
    c = SAPPHIRE
    if norm_type == "L2":
        _sphere(1.0, c)
    elif norm_type == "RMSNorm":
        _sphere(np.sqrt(3.0), c)
    elif norm_type == "LayerNorm":
        r = np.sqrt(3.0)
        th = np.linspace(0, 2 * np.pi, 160)
        u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
        v = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
        pts = r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
        (ln,) = ax.plot(pts[:,0], pts[:,1], pts[:,2], color=c, alpha=0.55, linewidth=1.65)
        constraint_artists.append(ln)
    elif norm_type == "L1":
        _octahedron(c)
    elif norm_type in ("DyT", "BatchNorm"):
        _cube(1.0, c)


# ====================== VECTOR DRAWING ======================
vector_lines = []
vectors = np.random.uniform(-AXIS_LIM, AXIS_LIM, (NUM_VECTORS, DIM))


def draw_vectors(norm_type):
    global vector_lines
    for ln in vector_lines:
        try:
            ln.remove()
        except (AttributeError, ValueError):
            pass
    vector_lines = []

    normed = apply_norm(vectors, norm_type)
    for i in range(NUM_VECTORS):
        (ln,) = ax.plot(
            [0, normed[i, 0]], [0, normed[i, 1]], [0, normed[i, 2]],
            color=COLORS[i], linewidth=3.0, solid_capstyle="round",
        )
        vector_lines.append(ln)


# ====================== FRAME SEQUENCE ======================
# Build a flat list of (norm_type, azimuth) for every frame we want to save.
raw_frames_per_norm = round(args.degrees_per_norm / AZIM_STEP_DEG)
gif_step = args.gif_step

frame_sequence = []  # list of (norm_type, azimuth_deg)
for norm in NORMS:
    for fi in range(raw_frames_per_norm):
        if fi % gif_step == 0:
            azim = START_AZIM - fi * AZIM_STEP_DEG
            frame_sequence.append((norm, azim))

total_frames = len(frame_sequence)


def make_frame(idx):
    norm_type, azim = frame_sequence[idx]

    # Only rebuild scene when norm changes (first frame or new norm).
    if idx == 0 or frame_sequence[idx - 1][0] != norm_type:
        draw_vectors(norm_type)
        draw_constraint_shape(norm_type)
        title_text.set_text(NORM_TITLES[norm_type])
        subtitle_text.set_text(NORM_SUBTITLES[norm_type])
        ax.view_init(elev=START_ELEV, azim=azim)
    else:
        ax.view_init(elev=START_ELEV, azim=azim)

    return []


# ====================== OUTPUT PATH ======================
if args.output:
    out_path = args.output
else:
    ts = datetime.datetime.now().strftime("%m_%d_%y_%H_%M_%S")
    out_dir = os.path.join("outputs", f"{ts}_norms_thumbnail")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "norms_thumbnail.gif")

# ====================== RENDER ======================
print(f"Rendering {total_frames} frames → {out_path}", flush=True)

anim = mplanimation.FuncAnimation(
    fig,
    make_frame,
    frames=total_frames,
    interval=INTERVAL_MS,
    blit=False,
    cache_frame_data=False,
)

gif_fps = round(1000 / INTERVAL_MS / gif_step)

with tqdm(total=total_frames, unit="frame") as pbar:
    anim.save(
        out_path,
        writer="pillow",
        fps=gif_fps,
        progress_callback=lambda i, n: pbar.update(1),
    )

print(f"Saved → {out_path}", flush=True)
