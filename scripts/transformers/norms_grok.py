import argparse
import datetime
import itertools
import os

from tqdm import tqdm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as mplanimation
from matplotlib import cm as mpl_cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons, CheckButtons, TextBox, Button
from matplotlib.patches import Rectangle

# ====================== CLI ======================
parser = argparse.ArgumentParser(description="3D normalization visualizer")
parser.add_argument(
    "--no-gif",
    action="store_true",
    help="Skip GIF generation and jump straight to the interactive plot.",
)
parser.add_argument(
    "--num-vectors",
    type=int,
    default=5,
    metavar="N",
    help="Number of random vectors to display (default: 5).",
)
parser.add_argument(
    "--gif-step",
    type=int,
    default=2,
    metavar="N",
    help="Save every Nth frame to the GIF (default: 2 = every other frame).",
)
args = parser.parse_args()

# ====================== CONFIG & DATA ======================
np.random.seed(42)
num_vectors = args.num_vectors
dim = 3

# Plot limits and max |coordinate| for unconstrained random endpoints.
AXIS_LIM = 2.0

# Sidebar / plot typography: match RadioButtons label size (rc "font.size", usually 10).
UI_FS = float(mpl.rcParams["font.size"])
VECTORS_TITLE_FS = UI_FS + 3

def sample_vectors(n, norm_type=None):
    return np.random.uniform(-AXIS_LIM, AXIS_LIM, (n, dim))


def constraint_indicator_flags():
    """Read-only (surface, inside) for *normalized* outputs under current_norm."""
    if current_norm in ("L2", "RMSNorm", "L1"):
        return True, False
    if current_norm == "LayerNorm":
        return True, False
    if current_norm == "DyT":
        return False, True
    if current_norm == "BatchNorm":
        return False, False
    return False, False


current_norm = "LayerNorm"
vectors = sample_vectors(num_vectors)

BASE_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]


def palette(n):
    if n <= len(BASE_COLORS):
        return BASE_COLORS[:n]
    out = list(BASE_COLORS)
    cmap = mpl_cm.get_cmap("tab10")
    for i in range(len(BASE_COLORS), n):
        out.append(cmap((i - len(BASE_COLORS)) % 10))
    return out


colors = palette(num_vectors)

# ====================== NORMALIZATION FUNCTIONS ======================
def apply_norm(X, norm_type, eps=1e-5):
    if norm_type == "LayerNorm":
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        return (X - mean) / (std + eps)

    elif norm_type == "BatchNorm":
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        return (X - mean) / (std + eps)

    elif norm_type == "DyT":
        alpha = 1.0
        return np.tanh(alpha * X)

    elif norm_type == "RMSNorm":
        # Per row: scale by RMS of features (no mean subtraction), like RMSNorm in transformers
        rms = np.sqrt(np.mean(X**2, axis=1, keepdims=True) + eps)
        return X / rms

    elif norm_type == "L1":
        d = np.sum(np.abs(X), axis=1, keepdims=True) + eps
        return X / d

    elif norm_type == "L2":
        # Per row: divide by Euclidean norm (points lie on the unit sphere)
        d = np.linalg.norm(X, axis=1, keepdims=True, ord=2) + eps
        return X / d

    return X


# ====================== MATPLOTLIB SETUP ======================
fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(111, projection="3d")

# Left column: narrow legend on top, widgets stacked below (no overlap).
_left_col_w = 0.26
_left_col_x = 0.02
plt.subplots_adjust(left=_left_col_x + _left_col_w + 0.04, right=0.95, bottom=0.08, top=0.92)

ax.set_xlim(-AXIS_LIM, AXIS_LIM)
ax.set_ylim(-AXIS_LIM, AXIS_LIM)
ax.set_zlim(-AXIS_LIM, AXIS_LIM)
ax.set_xlabel("X", fontsize=UI_FS)
ax.set_ylabel("Y", fontsize=UI_FS)
ax.set_zlabel("Z", fontsize=UI_FS)
ax.set_title("3D Vectors + Normalization", fontsize=UI_FS)
ax.tick_params(axis="both", labelsize=UI_FS)

# Store artists
norm_lines = []
orig_lines = []
constraint_artists = []

# Sapphire blue wireframes: where normalized outputs lie (per norm)
SAPPHIRE = "#0f4c99"

# Bordered panels (legend + normalized-output indicators)
_PANEL_FACE = "#f8f8f8"
_PANEL_EDGE = "#666666"
_PANEL_LW = 1.0


def _clear_constraint_shape():
    for a in constraint_artists:
        try:
            a.remove()
        except (AttributeError, ValueError):
            pass
    constraint_artists.clear()


def _sphere_wireframe(radius, color, alpha, lw):
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


def _octahedron_wireframe(color, alpha, lw):
    V = np.array(
        [
            [1.0, 0, 0],
            [-1.0, 0, 0],
            [0, 1.0, 0],
            [0, -1.0, 0],
            [0, 0, 1.0],
            [0, 0, -1.0],
        ]
    )
    for i in range(6):
        for j in range(i + 1, 6):
            if np.sum(np.abs(V[i] - V[j])) != 2.0:
                continue
            if np.dot(V[i], V[j]) != 0.0:
                continue
            (ln,) = ax.plot(
                [V[i, 0], V[j, 0]],
                [V[i, 1], V[j, 1]],
                [V[i, 2], V[j, 2]],
                color=color,
                alpha=alpha,
                linewidth=lw,
            )
            constraint_artists.append(ln)


def _cube_wireframe(half, color, alpha, lw):
    V = np.array(list(itertools.product([-half, half], repeat=3)), dtype=float)
    for i in range(8):
        for j in range(i + 1, 8):
            diff = V[i] - V[j]
            if np.count_nonzero(np.abs(diff) > 1e-9) != 1:
                continue
            (ln,) = ax.plot(
                [V[i, 0], V[j, 0]],
                [V[i, 1], V[j, 1]],
                [V[i, 2], V[j, 2]],
                color=color,
                alpha=alpha,
                linewidth=lw,
            )
            constraint_artists.append(ln)


def draw_constraint_shape():
    _clear_constraint_shape()
    alpha, lw = 0.42, 1.0
    c = SAPPHIRE

    if current_norm == "BatchNorm":
        # No hard compact surface: the true per-element bound is sqrt(B-1),
        # which grows with batch size and is extremely loose for random inputs.
        # By CLT (large B), each feature independently approaches N(0,1), so
        # we draw the ±2σ cube as an asymptotic approximation (~95% per feature).
        _cube_wireframe(1.0, c, alpha, lw)
        return

    if current_norm == "L2":
        _sphere_wireframe(1.0, c, alpha, lw)
    elif current_norm == "RMSNorm":
        _sphere_wireframe(np.sqrt(3.0), c, alpha, lw)
    elif current_norm == "LayerNorm":
        r = np.sqrt(3.0)
        th = np.linspace(0, 2 * np.pi, 160)
        u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
        v = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
        pts = r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
        (ln,) = ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=c, alpha=alpha + 0.08, linewidth=lw + 0.2)
        constraint_artists.append(ln)
    elif current_norm == "L1":
        _octahedron_wireframe(c, alpha, lw)
    elif current_norm == "DyT":
        _cube_wireframe(1.0, c, alpha, lw)


def constraint_legend_patch_and_caption():
    """(facecolor, math_text) for the side legend."""
    if current_norm == "L2":
        return SAPPHIRE, r"$L_2$: $\|\mathbf{v}\|_2 = 1$"
    if current_norm == "RMSNorm":
        return (
            SAPPHIRE,
            r"$\mathrm{RMS}$: $\frac{1}{d}\sum_i v_i^2 = 1$"
            "\n"
            r"$\Rightarrow \|\mathbf{v}\|_2 = \sqrt{d}$"
            + f"\n($d={dim}$)",
        )
    if current_norm == "LayerNorm":
        return (
            SAPPHIRE,
            r"$\mathrm{LN}$: $\sum_i v_i = 0,\ "
            r"\mathrm{Var}_{\mathrm{pop}}(\mathbf{v})=1$"
            "\n"
            r"$\Rightarrow \|\mathbf{v}\|_2 = \sqrt{d}$"
            + f"\n($d={dim}$)",
        )
    if current_norm == "L1":
        return SAPPHIRE, r"$L_1$: $\sum_i |v_i| = 1$"
    if current_norm == "DyT":
        return SAPPHIRE, r"$\mathrm{DyT}$: $y_i=\tanh(x_i)$" "\n" r"$y_i \in (-1,1)$"
    if current_norm == "BatchNorm":
        return (
            SAPPHIRE,
            r"$\mathrm{BN}$: by CLT (large $B$)," "\n"
            r"each feature $\approx\mathcal{N}(0,1)$" "\n"
            r"Cube $= \pm 1\sigma$ per feature",
        )
    return "#b0b0b0", ""


def _norm_output_locus_note():
    """Where normalized outputs live for the selected norm."""
    if current_norm == "L2":
        return "Normalized: on the sphere surface."
    if current_norm == "RMSNorm":
        return "Normalized: on the sphere surface."
    if current_norm == "LayerNorm":
        return "Normalized: on a circle (1D curve in R³)."
    if current_norm == "L1":
        return "Normalized: on the octahedron surface."
    if current_norm == "DyT":
        return "DyT outputs: strictly inside (-1,1)³ (interior)."
    if current_norm == "BatchNorm":
        return r"(approx.; hard bound $\sqrt{B-1}$ is loose)"
    return ""


def plot_vectors():
    global norm_lines, orig_lines

    for line in norm_lines + orig_lines:
        line.remove()
    norm_lines = []
    orig_lines = []

    normed = apply_norm(vectors, current_norm)

    for i in range(num_vectors):
        line, = ax.plot(
            [0, normed[i, 0]],
            [0, normed[i, 1]],
            [0, normed[i, 2]],
            color=colors[i],
            linewidth=3.5,
            solid_capstyle="round",
        )
        norm_lines.append(line)

    if show_original:
        for i in range(num_vectors):
            line, = ax.plot(
                [0, vectors[i, 0]],
                [0, vectors[i, 1]],
                [0, vectors[i, 2]],
                color=colors[i],
                linestyle="--",
                linewidth=1.5,
                alpha=0.45,
            )
            orig_lines.append(line)

    draw_constraint_shape()
    update_legend()
    draw_constraint_indicators()
    fig.canvas.draw_idle()


# ====================== LEGEND (top of left column) ======================
_legend_bottom, _legend_h = 0.61, 0.29
legend_ax = fig.add_axes([_left_col_x, _legend_bottom, _left_col_w, _legend_h])
legend_ax.axis("off")

_indicator_bottom, _indicator_h = 0.495, 0.105
indicator_ax = fig.add_axes(
    [_left_col_x, _indicator_bottom, _left_col_w, _indicator_h],
)
indicator_ax.axis("off")


def draw_constraint_indicators():
    indicator_ax.clear()
    indicator_ax.axis("off")
    indicator_ax.add_patch(
        Rectangle(
            (0.012, 0.02),
            0.976,
            0.93,
            transform=indicator_ax.transAxes,
            facecolor=_PANEL_FACE,
            edgecolor=_PANEL_EDGE,
            linewidth=_PANEL_LW,
            zorder=0,
        )
    )
    surface_on, inside_on = constraint_indicator_flags()
    chk, empty = "\u2611", "\u2610"
    indicator_ax.text(
        0.02,
        0.70,
        f"{chk if surface_on else empty}  Vectors Required to be on Volume Surface",
        transform=indicator_ax.transAxes,
        fontsize=UI_FS,
        va="center",
        color="#111",
        zorder=2,
    )
    indicator_ax.text(
        0.02,
        0.32,
        f"{chk if inside_on else empty}  Vectors Required to be Inside Volume",
        transform=indicator_ax.transAxes,
        fontsize=UI_FS,
        va="center",
        color="#111",
        zorder=2,
    )


def update_legend():
    legend_ax.clear()
    legend_ax.axis("off")
    legend_ax.add_patch(
        Rectangle(
            (0.012, 0.015),
            0.976,
            0.97,
            transform=legend_ax.transAxes,
            facecolor=_PANEL_FACE,
            edgecolor=_PANEL_EDGE,
            linewidth=_PANEL_LW,
            zorder=0,
        )
    )
    legend_ax.set_title(
        "Output constraint",
        fontsize=VECTORS_TITLE_FS,
        fontweight="600",
        pad=6,
        loc="left",
    )

    fc, math_cap = constraint_legend_patch_and_caption()
    if math_cap:
        sq = 0.22
        legend_ax.add_patch(
            Rectangle(
                (0.05, 0.58),
                sq,
                sq,
                transform=legend_ax.transAxes,
                facecolor=fc,
                edgecolor="black",
                linewidth=1.0,
                zorder=2,
            )
        )
        legend_ax.text(
            0.31,
            0.69,
            math_cap,
            transform=legend_ax.transAxes,
            fontsize=UI_FS,
            va="center",
            linespacing=1.45,
            zorder=3,
        )

    loc = _norm_output_locus_note()
    if loc:
        legend_ax.text(
            0.05,
            0.46,
            loc,
            transform=legend_ax.transAxes,
            fontsize=UI_FS,
            linespacing=1.35,
            color="#222",
            zorder=2,
        )
    legend_ax.text(
        0.05,
        0.30,
        "Raw vectors: uniform in the plot cube.",
        transform=legend_ax.transAxes,
        fontsize=UI_FS,
        linespacing=1.3,
        color="#333",
        zorder=2,
    )

    legend_ax.text(
        0.05,
        0.14,
        "Solid = normalized · Dashed = original",
        transform=legend_ax.transAxes,
        fontsize=UI_FS,
        color="#000000",
        zorder=2,
    )


# ====================== WIDGETS (below legend) ======================
show_original = True

_radio_bottom, _radio_h = 0.282, 0.208
_check_bottom, _check_h = 0.195, 0.08
_nvec_bottom, _nvec_h = 0.055, 0.085

radio_ax = fig.add_axes([_left_col_x, _radio_bottom, _left_col_w, _radio_h])
radio = RadioButtons(
    radio_ax,
    ["LayerNorm", "DyT", "BatchNorm", "RMSNorm", "L1", "L2"],
    active=0,
)
for _lbl in radio.labels:
    _lbl.set_fontsize(UI_FS)


def on_radio_change(label):
    global current_norm
    current_norm = label
    plot_vectors()


radio.on_clicked(on_radio_change)

check_ax = fig.add_axes([_left_col_x, _check_bottom, _left_col_w, _check_h])
check = CheckButtons(check_ax, ["Show Original Vectors"], [True])
for _lbl in check.labels:
    _lbl.set_fontsize(UI_FS)


def on_check_change(label):
    global show_original
    show_original = not show_original
    plot_vectors()


check.on_clicked(on_check_change)

nvec_ax = fig.add_axes([_left_col_x, _nvec_bottom, _left_col_w, _nvec_h])
text_box = TextBox(nvec_ax, "", initial=str(num_vectors))
if hasattr(text_box, "text_disp"):
    text_box.text_disp.set_fontsize(UI_FS)
nvec_ax.set_title(
    "Vectors", fontsize=VECTORS_TITLE_FS, pad=6, loc="left", fontweight="600"
)

NVEC_MIN, NVEC_MAX = 1, 60


def resize_vectors(n):
    global num_vectors, vectors, colors
    n = int(n)
    n = max(NVEC_MIN, min(n, NVEC_MAX))
    old_n = num_vectors
    if n == old_n:
        return n
    if n > old_n:
        extra = sample_vectors(n - old_n)
        vectors = np.vstack([vectors, extra])
    else:
        vectors = vectors[:n].copy()
    num_vectors = n
    colors = palette(n)
    return n


def on_nvec_submit(text):
    try:
        raw = int(float(str(text).strip()))
    except (TypeError, ValueError):
        text_box.set_val(str(num_vectors))
        return
    n = resize_vectors(raw)
    text_box.set_val(str(n))
    plot_vectors()
    fig.canvas.draw_idle()


text_box.on_submit(on_nvec_submit)

# ====================== SPIN (rotate view about z through origin) ======================
# Clockwise around +z as seen from above: step azimuth downward each frame.
_spin_active = False
_spin_anim = None
_SPIN_AZIM_STEP_DEG = 1.2
_SPIN_INTERVAL_MS = 35

spin_ax = fig.add_axes([0.86, 0.035, 0.09, 0.045])
spin_btn = Button(spin_ax, "Spin")
spin_btn.label.set_fontsize(UI_FS)


def _spin_frame(_frame):
    if not _spin_active:
        return []
    ax.view_init(elev=ax.elev, azim=ax.azim - _SPIN_AZIM_STEP_DEG)
    fig.canvas.draw_idle()
    return []


def on_spin_clicked(_event):
    global _spin_active, _spin_anim
    _spin_active = not _spin_active
    spin_btn.label.set_text("Stop" if _spin_active else "Spin")
    spin_btn.label.set_fontsize(UI_FS)
    if _spin_active:
        if _spin_anim is None:
            _spin_anim = mplanimation.FuncAnimation(
                fig,
                _spin_frame,
                interval=_SPIN_INTERVAL_MS,
                blit=False,
                cache_frame_data=False,
            )
        else:
            _spin_anim.event_source.start()
    elif _spin_anim is not None:
        _spin_anim.event_source.stop()


spin_btn.on_clicked(on_spin_clicked)

# ====================== GIF EXPORT ======================
def generate_gifs():
    global current_norm, show_original

    norms = ["LayerNorm", "DyT", "BatchNorm", "RMSNorm", "L1", "L2"]
    ts = datetime.datetime.now().strftime("%m_%d_%y_%H_%M_%S")
    out_dir = os.path.join("outputs", f"{ts}_Norm_GIFS")
    os.makedirs(out_dir, exist_ok=True)

    frames_per_rotation = round(360 / _SPIN_AZIM_STEP_DEG)
    fps = round(1000 / _SPIN_INTERVAL_MS)

    total = len(norms)
    total_frames = frames_per_rotation * 2  # rotation 1: with orig, rotation 2: without
    gif_step = args.gif_step
    gif_fps = round(fps / gif_step)

    for done, norm in enumerate(norms, start=1):
        # Ensure checkbox is checked (show_original=True) before each GIF.
        if not check.get_status()[0]:
            check.set_active(0)   # toggles to checked; callback sets show_original=True
        # Drive norm through the radio widget so the visual and current_norm stay in sync.
        radio.set_active(norms.index(norm))   # fires on_radio_change → sets current_norm + plot_vectors
        ax.view_init(elev=28, azim=-50)

        # Flag reset per norm so the halfway switch fires exactly once per GIF.
        switched = [False]

        def make_frame(frame_idx):
            if frame_idx >= frames_per_rotation and not switched[0]:
                switched[0] = True
                check.set_active(0)   # toggles to unchecked; callback sets show_original=False
            ax.view_init(
                elev=28,
                azim=-50 - (frame_idx % frames_per_rotation) * _SPIN_AZIM_STEP_DEG,
            )
            return []

        gif_frames = range(0, total_frames, gif_step)
        anim = mplanimation.FuncAnimation(
            fig,
            make_frame,
            frames=gif_frames,
            interval=_SPIN_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )

        path = os.path.join(out_dir, f"{norm}.gif")
        print(f"[{done}/{total}] Generating {path} ...", flush=True)
        with tqdm(total=len(gif_frames), unit="frame", leave=False) as frame_pbar:
            anim.save(
                path,
                writer="pillow",
                fps=gif_fps,
                progress_callback=lambda i, n: frame_pbar.update(1),
            )
        print(f"       saved → {path}", flush=True)


# ====================== INITIAL PLOT ======================
plot_vectors()
ax.view_init(elev=28, azim=-50)

if not args.no_gif:
    generate_gifs()
    # Restore default state: show_original global was left False by the last GIF
    # iteration; the checkbox widget visual is still checked=True (we never
    # touched it during generation), so setting show_original=True re-syncs them.
    current_norm = "LayerNorm"
    check.set_active(0)   # checkbox is unchecked after last GIF; toggle back to checked
    radio.set_active(0)   # fires on_radio_change → sets current_norm + plot_vectors
    ax.view_init(elev=28, azim=-50)
    print("All GIFs saved to outputs/. Interactive plot is now active.", flush=True)

plt.show()
