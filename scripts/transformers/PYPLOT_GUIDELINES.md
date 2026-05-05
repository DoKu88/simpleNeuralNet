# PyPlot Applet Guidelines

These guidelines define how to write Python visualization code for the blog's interactive applet system.

## Runtime Environment

Code runs inside **PyScript 2024.11.1** in a sandboxed `<iframe>`. The page pre-loads **Plotly.js 2.35.2**, which is available in the browser JS context. The output renders into a `<div id="pyscript-output">`.

There is no server-side Python — everything executes client-side in the browser via WebAssembly.

## Required Imports

```python
from js import Plotly
from pyscript.ffi import to_js
```

- `Plotly` is the Plotly.js object exposed from the browser. Use it to render charts.
- `to_js` converts Python dicts/lists into JS objects before passing to Plotly.

## Rendering a Plot

Always call `Plotly.newPlot` with the target div ID `"pyscript-output"`:

```python
Plotly.newPlot("pyscript-output", to_js(data), to_js(layout))
```

Both `data` and `layout` must be converted with `to_js()`. NumPy arrays must be converted to plain Python lists first with `.tolist()`.

## Layout Conventions

- Set `"height"` explicitly — around `460` for standard plots, `520` for tall/3D.
- Keep margins tight for mobile: `"margin": {"l": 40, "r": 20, "t": 40, "b": 40}`.
- Do **not** set `"autosize"` or `"responsive"` — the iframe host injects these automatically.
- Do **not** set `config` (e.g., `scrollZoom`) — the host intercepts `Plotly.newPlot` and applies `responsive: true, scrollZoom: true` before your call resolves.
- Titles should be short and descriptive. Include interaction hints when relevant (e.g., "drag to spin, scroll to zoom").

## Available Packages (no install needed)

These are available by default and should be listed in the `packages` config when used:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib` (avoid — see note below)
- `micropip`

Any pure-Python PyPI package can be installed at runtime:

```python
import micropip
await micropip.install('somepackage')
```

## Matplotlib vs Plotly

**Prefer Plotly for all interactive figures.** Matplotlib renders to a static canvas and requires extra work to display (base64 PNG injection). Plotly gives interactive zoom, hover, and 3D rotation out of the box, which is the point of these applets.

Only use matplotlib if the visualization is genuinely better as a static image and interactivity adds no value (rare).

## Minimal Working Example

```python
import numpy as np
from js import Plotly
from pyscript.ffi import to_js

x = np.linspace(0, 2 * np.pi, 200).tolist()
y = np.sin(x)

data = [{"type": "scatter", "x": x, "y": y, "mode": "lines", "name": "sin(x)"}]
layout = {
    "title": "Sine Wave",
    "height": 400,
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    "xaxis": {"title": "x"},
    "yaxis": {"title": "y"},
}

Plotly.newPlot("pyscript-output", to_js(data), to_js(layout))
```

## 3D Example

```python
import numpy as np
from js import Plotly
from pyscript.ffi import to_js

x = np.linspace(-5, 5, 60).tolist()
y = np.linspace(-5, 5, 60).tolist()
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)).tolist()

data = [{"type": "surface", "z": Z, "x": x, "y": y, "colorscale": "Viridis"}]
layout = {
    "title": "3D Wave — drag to spin, scroll to zoom",
    "height": 460,
    "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
}

Plotly.newPlot("pyscript-output", data=to_js(data), layout=to_js(layout))
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| Passing a numpy array directly to `to_js` | Call `.tolist()` first |
| Setting `autosize`, `responsive`, or `scrollZoom` | Remove them — injected by the host |
| Using `plt.show()` from matplotlib | Use `Plotly.newPlot` instead |
| Targeting a div ID other than `"pyscript-output"` | Always use `"pyscript-output"` |
| Using `async def` / `await` at top level without micropip | Only needed when calling `micropip.install` |

## Package Declaration

When the code uses packages beyond the defaults, list them as a comma-separated string in the packages field (e.g. `plotly,numpy,scipy`). The runtime installs them before executing the script.

`plotly` and `numpy` are the most common combination and are the default.
