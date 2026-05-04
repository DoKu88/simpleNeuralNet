import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mplanimation
from matplotlib import cm as mpl_cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons, CheckButtons, TextBox, Slider, Button
from matplotlib.patches import Rectangle

# ====================== CONFIG & DATA ======================
np.random.seed(42)
num_vectors = 5
dim = 3

vectors = np.random.randn(num_vectors, dim) * 2.5

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

# Legend: max rows visible; scroll when num_vectors exceeds this
LEGEND_MAX_ROWS = 6
legend_scroll = 0
scroll_slider = None

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

    return X


# ====================== MATPLOTLIB SETUP ======================
fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(111, projection="3d")

# Left column: narrow legend on top, widgets stacked below (no overlap).
_left_col_w = 0.23
_left_col_x = 0.02
plt.subplots_adjust(left=_left_col_x + _left_col_w + 0.04, right=0.95, bottom=0.08, top=0.92)

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(-6, 6)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Vectors + Normalization Effects")

# Store artists
norm_lines = []
orig_lines = []


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

    fig.canvas.draw_idle()


# ====================== LEGEND (top of left column) ======================
_legend_bottom, _legend_h = 0.50, 0.32
legend_ax = fig.add_axes([_left_col_x, _legend_bottom, _left_col_w, _legend_h])
legend_ax.axis("off")
legend_ax.set_title(
    "Vector legend (color + coords)", fontsize=9, pad=6, loc="left"
)

_scroll_bottom, _scroll_h = 0.465, 0.028
scroll_ax = fig.add_axes([_left_col_x, _scroll_bottom, _left_col_w, _scroll_h])

_row = 0.105
_sw = 0.065
_sh = 0.065


def update_legend():
    legend_ax.clear()
    legend_ax.axis("off")
    legend_ax.set_title(
        "Vector legend (color + coords)", fontsize=9, pad=6, loc="left"
    )

    first = legend_scroll
    last = min(num_vectors, first + LEGEND_MAX_ROWS)
    for j, i in enumerate(range(first, last)):
        y = 0.90 - j * _row
        rect = Rectangle(
            (0.02, y - 0.5 * _sh),
            _sw,
            _sh,
            facecolor=colors[i],
            edgecolor="black",
            linewidth=0.6,
            transform=legend_ax.transAxes,
        )
        legend_ax.add_artist(rect)

        coords = f"v{i+1}: ({vectors[i,0]:.2f}, {vectors[i,1]:.2f}, {vectors[i,2]:.2f})"
        legend_ax.text(
            0.11,
            y,
            coords,
            va="center",
            fontsize=7,
            fontfamily="monospace",
        )

    if num_vectors > LEGEND_MAX_ROWS:
        legend_ax.text(
            0.02,
            0.04,
            f"Rows {first + 1}–{last} of {num_vectors} · use slider below",
            fontsize=6.5,
            style="italic",
            color="gray",
        )
    else:
        legend_ax.text(
            0.02,
            0.04,
            "Solid = normalized · Dashed = original",
            fontsize=7,
            style="italic",
            color="gray",
        )


def _on_scroll_change(val):
    global legend_scroll
    legend_scroll = int(round(val))
    update_legend()
    fig.canvas.draw_idle()


def setup_legend_scroll_slider():
    global scroll_slider, legend_scroll
    max_scroll = max(0, num_vectors - LEGEND_MAX_ROWS)
    legend_scroll = min(legend_scroll, max_scroll)

    scroll_ax.clear()
    if max_scroll == 0:
        scroll_slider = None
        scroll_ax.set_visible(False)
        scroll_ax.axis("off")
        return

    scroll_ax.set_visible(True)
    # Empty widget label — Slider draws it to the left of a thin axes and it clips
    scroll_slider = Slider(
        scroll_ax,
        "",
        0,
        max_scroll,
        valinit=legend_scroll,
        valstep=1,
        valfmt="%d",
    )
    scroll_ax.set_title("Scroll", fontsize=8, pad=2, loc="left")
    scroll_slider.on_changed(_on_scroll_change)


# ====================== WIDGETS (below legend) ======================
current_norm = "LayerNorm"
show_original = True

_radio_bottom, _radio_h = 0.21, 0.22
_check_bottom, _check_h = 0.09, 0.10
_nvec_bottom, _nvec_h = 0.02, 0.055

radio_ax = fig.add_axes([_left_col_x, _radio_bottom, _left_col_w, _radio_h])
radio = RadioButtons(radio_ax, ["LayerNorm", "DyT", "BatchNorm"], active=0)


def on_radio_change(label):
    global current_norm
    current_norm = label
    plot_vectors()


radio.on_clicked(on_radio_change)

check_ax = fig.add_axes([_left_col_x, _check_bottom, _left_col_w, _check_h])
check = CheckButtons(check_ax, ["Show Original Vectors"], [True])


def on_check_change(label):
    global show_original
    show_original = not show_original
    plot_vectors()


check.on_clicked(on_check_change)

nvec_ax = fig.add_axes([_left_col_x, _nvec_bottom, _left_col_w, _nvec_h])
text_box = TextBox(nvec_ax, "", initial=str(num_vectors))
nvec_ax.set_title("Vectors", fontsize=8, pad=2, loc="left")

NVEC_MIN, NVEC_MAX = 1, 60


def resize_vectors(n):
    global num_vectors, vectors, colors, legend_scroll
    n = int(n)
    n = max(NVEC_MIN, min(n, NVEC_MAX))
    old_n = num_vectors
    if n == old_n:
        return n
    if n > old_n:
        extra = np.random.randn(n - old_n, dim) * 2.5
        vectors = np.vstack([vectors, extra])
    else:
        vectors = vectors[:n].copy()
    num_vectors = n
    colors = palette(n)
    legend_scroll = min(legend_scroll, max(0, n - LEGEND_MAX_ROWS))
    return n


def on_nvec_submit(text):
    global legend_scroll
    try:
        raw = int(float(str(text).strip()))
    except (TypeError, ValueError):
        text_box.set_val(str(num_vectors))
        return
    n = resize_vectors(raw)
    text_box.set_val(str(n))
    setup_legend_scroll_slider()
    update_legend()
    plot_vectors()
    fig.canvas.draw_idle()


text_box.on_submit(on_nvec_submit)

setup_legend_scroll_slider()
update_legend()

# ====================== SPIN (rotate view about z through origin) ======================
# Clockwise around +z as seen from above: step azimuth downward each frame.
_spin_active = False
_spin_anim = None
_SPIN_AZIM_STEP_DEG = 1.2
_SPIN_INTERVAL_MS = 35

spin_ax = fig.add_axes([0.86, 0.035, 0.09, 0.045])
spin_btn = Button(spin_ax, "Spin")


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

# ====================== INITIAL PLOT ======================
plot_vectors()
ax.view_init(elev=28, azim=-50)

plt.show()
