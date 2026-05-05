import itertools
import numpy as np
from js import Plotly, document
from pyscript.ffi import to_js, create_proxy

# ── constants ──────────────────────────────────────────────────────────────────
AXIS_LIM = 2.0
DIM = 3
MAX_VECTORS = 60
SAPPHIRE = "#0f4c99"
BASE_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
    "#CC79A7", "#D55E00", "#999999", "#44AA99", "#882255",
]
NORMS = ["LayerNorm", "DyT", "BatchNorm", "RMSNorm", "L1", "L2"]

CONSTRAINT_CAPTIONS = {
    "L2":        "‖v‖₂ = 1  →  unit sphere",
    "RMSNorm":   f"RMS(v)=1  →  sphere, ‖v‖₂=√{DIM}≈{np.sqrt(DIM):.3f}",
    "LayerNorm": f"Σvᵢ=0, Var=1  →  circle in R³, ‖v‖₂=√{DIM}≈{np.sqrt(DIM):.3f}",
    "L1":        "Σ|vᵢ|=1  →  octahedron surface",
    "DyT":       "yᵢ=tanh(αxᵢ)  →  strictly inside (−1,1)³",
    "BatchNorm": "per-feature ≈ N(0,1) by CLT  →  inside ±1σ cube",
}

# ── state ──────────────────────────────────────────────────────────────────────
np.random.seed(42)
_pool = np.random.uniform(-AXIS_LIM, AXIS_LIM, (MAX_VECTORS, DIM))

state = {"norm": "LayerNorm", "show_original": True, "n": 5}

# ── helpers ────────────────────────────────────────────────────────────────────
def get_vecs():
    return _pool[:state["n"]]

def get_colors():
    n = state["n"]
    if n <= len(BASE_COLORS):
        return BASE_COLORS[:n]
    extra = [f"hsl({(i * 37) % 360},65%,50%)" for i in range(len(BASE_COLORS), n)]
    return BASE_COLORS + extra

# ── normalization ──────────────────────────────────────────────────────────────
def apply_norm(X, norm_type, eps=1e-5):
    if norm_type == "LayerNorm":
        mu = np.mean(X, axis=1, keepdims=True)
        sigma = np.std(X, axis=1, keepdims=True)
        return (X - mu) / (sigma + eps)
    if norm_type == "BatchNorm":
        mu = np.mean(X, axis=0, keepdims=True)
        sigma = np.std(X, axis=0, keepdims=True)
        return (X - mu) / (sigma + eps)
    if norm_type == "DyT":
        return np.tanh(X)
    if norm_type == "RMSNorm":
        rms = np.sqrt(np.mean(X ** 2, axis=1, keepdims=True) + eps)
        return X / rms
    if norm_type == "L1":
        d = np.sum(np.abs(X), axis=1, keepdims=True) + eps
        return X / d
    if norm_type == "L2":
        d = np.linalg.norm(X, axis=1, keepdims=True) + eps
        return X / d
    return X

# ── constraint wireframe builders ──────────────────────────────────────────────
# Each returns (xs, ys, zs) plain Python lists with None as line-break separators.

def _seg(a, b, xs, ys, zs):
    xs += [float(a[0]), float(b[0]), None]
    ys += [float(a[1]), float(b[1]), None]
    zs += [float(a[2]), float(b[2]), None]

def sphere_wire(r):
    xs, ys, zs = [], [], []
    phi = np.linspace(0, 2 * np.pi, 40)
    for t in np.linspace(0.12, np.pi - 0.12, 8):
        xs += (r * np.sin(t) * np.cos(phi)).tolist() + [None]
        ys += (r * np.sin(t) * np.sin(phi)).tolist() + [None]
        zs += (r * np.cos(t) * np.ones(40)).tolist() + [None]
    thetas = np.linspace(0, np.pi, 24)
    for p in phi[::5]:
        xs += (r * np.sin(thetas) * np.cos(p)).tolist() + [None]
        ys += (r * np.sin(thetas) * np.sin(p)).tolist() + [None]
        zs += (r * np.cos(thetas)).tolist() + [None]
    return xs, ys, zs

def layernorm_circle_wire():
    r = np.sqrt(float(DIM))
    th = np.linspace(0, 2 * np.pi, 160)
    u = np.array([1., -1., 0.]) / np.sqrt(2.)
    v = np.array([1., 1., -2.]) / np.sqrt(6.)
    pts = r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
    return pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist()

def octahedron_wire():
    V = np.array(
        [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float
    )
    xs, ys, zs = [], [], []
    for i in range(6):
        for j in range(i + 1, 6):
            if abs(np.sum(np.abs(V[i] - V[j])) - 2.) > 1e-9:
                continue
            if abs(np.dot(V[i], V[j])) > 1e-9:
                continue
            _seg(V[i], V[j], xs, ys, zs)
    return xs, ys, zs

def cube_wire(half):
    V = list(itertools.product([-half, half], repeat=3))
    xs, ys, zs = [], [], []
    for i in range(8):
        for j in range(i + 1, 8):
            diff = [V[i][k] - V[j][k] for k in range(3)]
            if sum(1 for d in diff if abs(d) > 1e-9) != 1:
                continue
            _seg(V[i], V[j], xs, ys, zs)
    return xs, ys, zs

def constraint_wire():
    norm = state["norm"]
    if norm == "L2":
        return sphere_wire(1.)
    if norm == "RMSNorm":
        return sphere_wire(np.sqrt(float(DIM)))
    if norm == "LayerNorm":
        return layernorm_circle_wire()
    if norm == "L1":
        return octahedron_wire()
    return cube_wire(1.)  # DyT, BatchNorm

# ── trace / layout builders ────────────────────────────────────────────────────
def build_traces():
    vecs = get_vecs()
    normed = apply_norm(vecs, state["norm"])
    colors = get_colors()
    n = state["n"]

    xs, ys, zs = constraint_wire()
    traces = [{
        "type": "scatter3d",
        "x": xs, "y": ys, "z": zs,
        "mode": "lines",
        "line": {"color": SAPPHIRE, "width": 2},
        "opacity": 0.45,
        "hoverinfo": "skip",
        "showlegend": False,
    }]

    for i in range(n):
        traces.append({
            "type": "scatter3d",
            "x": [0., float(normed[i, 0])],
            "y": [0., float(normed[i, 1])],
            "z": [0., float(normed[i, 2])],
            "mode": "lines+markers",
            "line": {"color": colors[i], "width": 6},
            "marker": {"size": [0, 5], "color": colors[i]},
            "hoverinfo": "skip",
            "showlegend": False,
        })

    if state["show_original"]:
        for i in range(n):
            traces.append({
                "type": "scatter3d",
                "x": [0., float(vecs[i, 0])],
                "y": [0., float(vecs[i, 1])],
                "z": [0., float(vecs[i, 2])],
                "mode": "lines",
                "line": {"color": colors[i], "width": 2, "dash": "dash"},
                "opacity": 0.40,
                "hoverinfo": "skip",
                "showlegend": False,
            })

    return traces

def build_layout():
    norm = state["norm"]
    caption = CONSTRAINT_CAPTIONS.get(norm, "")
    return {
        "scene": {
            "xaxis": {"range": [-AXIS_LIM, AXIS_LIM], "title": "X"},
            "yaxis": {"range": [-AXIS_LIM, AXIS_LIM], "title": "Y"},
            "zaxis": {"range": [-AXIS_LIM, AXIS_LIM], "title": "Z"},
            "aspectmode": "cube",
        },
        "title": {
            "text": (
                f"{norm}  —  drag to spin, scroll to zoom"
                f"<br><sup>{caption}  ·  solid = normalized, dashed = original</sup>"
            ),
            "font": {"size": 13},
            "x": 0.5,
        },
        "height": 500,
        "margin": {"l": 0, "r": 0, "t": 70, "b": 0},
        "showlegend": False,
        "paper_bgcolor": "white",
    }

# ── DOM setup ──────────────────────────────────────────────────────────────────
output = document.getElementById("pyscript-output")
output.innerHTML = """
<div id="norm-wrap" style="font-family:sans-serif;font-size:13px;">
  <div id="norm-controls" style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:8px 2px 6px;">
    <strong style="margin-right:2px;">Norm:</strong>
    <button class="nb active-nb" data-norm="LayerNorm">LayerNorm</button>
    <button class="nb" data-norm="DyT">DyT</button>
    <button class="nb" data-norm="BatchNorm">BatchNorm</button>
    <button class="nb" data-norm="RMSNorm">RMSNorm</button>
    <button class="nb" data-norm="L1">L1</button>
    <button class="nb" data-norm="L2">L2</button>
    <label style="margin-left:12px;cursor:pointer;">
      <input type="checkbox" id="show-orig" checked>
      Show original vectors
    </label>
    <span style="margin-left:12px;">
      Vectors:
      <input type="number" id="nvec" value="5" min="1" max="60"
             style="width:48px;padding:2px 4px;border:1px solid #aaa;border-radius:3px;">
    </span>
  </div>
  <div id="norm-plot"></div>
</div>
"""

_style = document.createElement("style")
_style.textContent = (
    ".nb{padding:3px 10px;border:1px solid #aaa;border-radius:3px;"
    "background:#f0f0f0;cursor:pointer;font-size:13px;}"
    ".active-nb{background:#0f4c99 !important;color:#fff !important;"
    "border-color:#0f4c99 !important;}"
)
document.head.appendChild(_style)

# ── render ─────────────────────────────────────────────────────────────────────
_initialized = False

def render():
    global _initialized
    traces = build_traces()
    layout = build_layout()
    if not _initialized:
        Plotly.newPlot("norm-plot", to_js(traces), to_js(layout))
        _initialized = True
    else:
        Plotly.react("norm-plot", to_js(traces), to_js(layout))

# ── event handlers ─────────────────────────────────────────────────────────────
def on_norm_click(e):
    norm = e.target.getAttribute("data-norm")
    if not norm:
        return
    state["norm"] = norm
    btns = document.querySelectorAll(".nb")
    for idx in range(btns.length):
        btns.item(idx).classList.remove("active-nb")
    e.target.classList.add("active-nb")
    render()

def on_show_orig(e):
    state["show_original"] = bool(e.target.checked)
    render()

def on_nvec(e):
    try:
        n = max(1, min(MAX_VECTORS, int(e.target.value)))
        e.target.value = str(n)
        state["n"] = n
        render()
    except Exception:
        pass

document.getElementById("norm-controls").addEventListener(
    "click", create_proxy(on_norm_click)
)
document.getElementById("show-orig").addEventListener(
    "change", create_proxy(on_show_orig)
)
document.getElementById("nvec").addEventListener(
    "change", create_proxy(on_nvec)
)

render()
