"""
MLP built from node classes. Each node tracks inputs, has a forward equation,
and a layer-specific backprop. Uses an abstract base class with concrete
implementations: LinearLayer, activations, and Loss.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from abc import ABC, abstractmethod
from datetime import datetime
from numpy.lib.stride_tricks import sliding_window_view
import petname

# Run identifier for saved plots: [timestamp]_[RUN_NAME]_[plot_type].png
RUN_NAME = petname.Generate(2, "_")  # e.g. wiggly_yellowtail
OUTPUT_DIR = "outputs"

# -----------------------------------------------------------------------------
# Abstract base: all nodes track inputs and define forward + backward
# -----------------------------------------------------------------------------

class Node(ABC):
    """Abstract base for MLP nodes. Tracks inputs for backprop."""

    def __init__(self):
        self._inputs = []  # track inputs passed through this node

    def _record_input(self, x: np.ndarray) -> None:
        self._inputs.append(x.copy())

    @property
    @abstractmethod
    def equation(self) -> str:
        """Human-readable equation for this layer, e.g. 'y = Mx + b'."""
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute output; must record input for backward."""
        pass

    @abstractmethod
    def backward(self, upstream: np.ndarray) -> np.ndarray:
        """Given dL/d(output), return dL/d(input). May update params (e.g. W, b)."""
        pass

    def update_weights(self, lr: float) -> None:
        """Gradient descent step. No-op for nodes without parameters (e.g. activation, loss)."""
        pass

# -----------------------------------------------------------------------------
# Non-linear activations (abstract activation + concrete ReLU, Sigmoid)
# -----------------------------------------------------------------------------

class Activation(Node, ABC):
    """Base for element-wise activation layers. Equation is subclass-specific."""

    def __init__(self):
        super().__init__()
        self._last_output = None  # often needed for backward (e.g. sigmoid'(y)=y*(1-y))

    @abstractmethod
    def _f(self, x: np.ndarray) -> np.ndarray:
        """Element-wise activation."""
        pass

    @abstractmethod
    def _f_prime(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Derivative w.r.t. input; may use stored output y = f(x)."""
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self._record_input(x)
        y = self._f(x)
        self._last_output = y
        return y

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        x = self._inputs[-1]
        y = self._last_output
        return upstream * self._f_prime(x, y)

# -----------------------------------------------------------------------------
# Linear layer: y = Mx + b
# -----------------------------------------------------------------------------

class LinearLayer(Node):
    """Layer equation: y = W @ x + b. Tracks x for backprop; updates W, b in backward."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None):
        super().__init__()
        rng = rng or np.random.default_rng()
        # Xavier (Glorot) initialization: scale by 1/sqrt(fan_in + fan_out)
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = rng.standard_normal((out_features, in_features)) * scale
        self.b = np.zeros(out_features)
        self._last_x = None  # last input for this forward pass

    @property
    def equation(self) -> str:
        return "y = W @ x + b"

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self._record_input(x)
        self._last_x = x
        return self.W @ x + self.b

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        # upstream = dL/dy  shape (out_features,)
        upstream = np.asarray(upstream, dtype=float)
        x = self._inputs[-1]  # same as _last_x

        # dL/db = dL/dy (since y = Wx + b implies dy/db = 1 i.e I)
        self.dLdb = upstream.copy()

        # dL/dW: Forward has y_i = sum_j W_ij x_j + b_i i.e. W_i: * x_j + b_i, so dy_i/dW_ij = x_j.
        # Chain rule: dL/dW_ij = (dL/dy_i) * (dy_i/dW_ij) = (dL/dy_i) * x_j.
        # So the (i,j) entry of dLdW is upstream[i] * x[j] = np.outer(upstream, x).
        self.dLdW = np.outer(upstream, x)

        # dL/dx = W.T @ (dL/dy)
        dx = self.W.T @ upstream
        return dx

    def update_weights(self, lr: float) -> None:
        """Gradient descent step: W -= lr * dLdW, b -= lr * dLdb. Call after backward."""
        self.W -= lr * self.dLdW
        self.b -= lr * self.dLdb

class Convolution2DLayer(Node):
  """2D Convolution (single channel, single kernel) using an im2col implementation."""

  def __init__(
      self,
      kernel_size: np.ndarray,
      in_features: np.ndarray,
      out_features: np.ndarray,
      padding_x: int,
      padding_y: int,
  ):
    # worry about stride and dilation later
    super().__init__()

    # Kernel shape and padding
    k_h, k_w = int(kernel_size[0]), int(kernel_size[1])
    self.kernel_size = (k_h, k_w)
    self.padding = (int(padding_y), int(padding_x))  # (pad_h, pad_w)

    # Xavier (Glorot) initialization for a single 2D kernel of shape (k_h, k_w)
    fan_in = k_h * k_w
    fan_out = k_h * k_w
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    rng = np.random.default_rng()
    self.kernel = rng.standard_normal((k_h, k_w)) * scale

    # Bias term for this feature map (scalar added to every output pixel)
    self.bias = 0.0

    # Store meta (not yet used in computation, but keep for API symmetry)
    self.padding_x = padding_x
    self.padding_y = padding_y
    self.in_features = in_features
    self.out_features = out_features

    # Caches for backward
    self._last_input: np.ndarray | None = None
    self._last_input_shape: tuple[int, int] | None = None
    self._last_cols: np.ndarray | None = None
    self._last_output_shape: tuple[int, int] | None = None
    self.dLdK: np.ndarray | None = None
    self.dLdbias: float | None = None

  @property
  def equation(self) -> str:
    return (
        "o[p, q] = sum_{a=0}^{kw-1} sum_{b=0}^{kh-1} "
        "k[a, b] * z[p - a + P1, q - b + P2]"
    )

  def _im2col(
      self,
      z_input: np.ndarray,
      kernel_size: tuple[int, int],
      padding: tuple[int, int],
  ) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Turn input z_input into a (num_patches, k_h * k_w) matrix using im2col.

    Returns:
      cols: (num_patches, k_h * k_w)
      out_shape: (out_h, out_w) spatial shape of the convolution output
    """
    k_h, k_w = kernel_size
    pad_h, pad_w = padding

    if pad_h or pad_w:
      z_input = np.pad(z_input, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    H, W = z_input.shape
    out_h = H - k_h + 1
    out_w = W - k_w + 1

    # (out_h, out_w, k_h, k_w)
    # sliding_window_view() creates a view into the input array where each element is a window of shape (k_h, k_w).
    # This is used to efficiently extract all sliding kernel-sized patches from the padded input image,
    # allowing us to flatten and arrange them column-wise for matrix multiplication with the kernel.
    patches = sliding_window_view(z_input, (k_h, k_w))
    cols = patches.reshape(out_h * out_w, k_h * k_w)
    return cols, (out_h, out_w)

  def _convolve2d_im2col(
      self,
      z_input: np.ndarray,
      kernel: np.ndarray,
      padding: tuple[int, int],
  ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Core 2D convolution (technically correlation) using im2col.

    Args:
      z_input: (H, W) input
      kernel: (k_h, k_w)
      padding: (pad_h, pad_w)

    Returns:
      out: (out_h, out_w)
      cols: im2col matrix (num_patches, k_h * k_w)
      out_shape: (out_h, out_w)
    """
    cols, out_shape = self._im2col(z_input, kernel.shape, padding)
    kernel_flat = kernel.reshape(-1)        # (k_h * k_w,)
    out_flat = cols @ kernel_flat           # (num_patches,)
    out = out_flat.reshape(out_shape)
    return out, cols, out_shape

  def forward(self, z_input: np.ndarray) -> np.ndarray:
    """
    Forward 2D convolution for a single-channel input and single kernel.
    z_input: (H, W)
    """
    z_input = np.asarray(z_input, dtype=float)
    self._record_input(z_input)

    out, cols, out_shape = self._convolve2d_im2col(z_input, self.kernel, self.padding)

    # Add bias term (broadcast over all spatial locations)
    out = out + self.bias

    # Cache for backward
    self._last_input = z_input
    self._last_input_shape = z_input.shape
    self._last_cols = cols
    self._last_output_shape = out_shape

    return out

  def backward(self, upstream: np.ndarray) -> np.ndarray:
    """
    Backward pass for 2D convolution.

    Given upstream gradient dL/dout (same shape as forward output),
    compute:
      - dL/dkernel (stored in self.dLdK)
      - dL/dz (returned)
    """
    upstream = np.asarray(upstream, dtype=float)

    if self._last_cols is None or self._last_input_shape is None or self._last_output_shape is None:
      raise RuntimeError("Convolution2DLayer.backward called before forward or cache missing.")

    cols = self._last_cols
    k_h, k_w = self.kernel.shape

    # ---- Gradient w.r.t. kernel ----
    # cols: (num_patches, k_h * k_w)
    # upstream_flat: (num_patches,)
    upstream_flat = upstream.reshape(-1)
    dK_flat = cols.T @ upstream_flat  # (k_h * k_w,)
    self.dLdK = dK_flat.reshape(k_h, k_w)

    # ---- Gradient w.r.t. bias ----
    # bias is added uniformly to every output pixel: dL/dbias = sum_{p,q} upstream[p,q]
    self.dLdbias = float(np.sum(upstream))

    # ---- Gradient w.r.t. input z ----
    # Use convolution of upstream with flipped kernel, then crop padding.
    pad_h, pad_w = self.padding

    # Flip kernel for gradient wrt input
    k_flipped = np.flip(self.kernel)

    # Full convolution of upstream with flipped kernel
    full_cols, full_shape = self._im2col(upstream, k_flipped.shape, (k_h - 1, k_w - 1))
    k_flat = k_flipped.reshape(-1)
    full_flat = full_cols @ k_flat
    full = full_flat.reshape(full_shape)  # gradient wrt padded input

    H_in, W_in = self._last_input_shape
    H_p = H_in + 2 * pad_h
    W_p = W_in + 2 * pad_w

    if full.shape != (H_p, W_p):
      raise RuntimeError(f"Unexpected full gradient shape {full.shape}, expected {(H_p, W_p)}.")

    if pad_h or pad_w:
      dz = full[pad_h:H_p - pad_h, pad_w:W_p - pad_w]
    else:
      dz = full

    return dz

  def update_weights(self, lr: float) -> None:
    """Gradient descent step on the kernel and bias: k -= lr * dLdK, b -= lr * dLdbias."""
    if self.dLdK is None or self.dLdbias is None:
      raise RuntimeError("update_weights called before backward; gradients are None.")
    self.kernel -= lr * self.dLdK
    self.bias -= lr * self.dLdbias


class ReLU(Activation):
    """ReLU: y = max(0, x). Backprop: pass upstream where x > 0."""

    @property
    def equation(self) -> str:
        return "y = max(0, x)"

    def _f(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _f_prime(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class LeakyReLU(Activation):
    """LeakyReLU: y = x if x > 0 else alpha * x.
    Prevents dying neurons by keeping a small gradient for negative inputs."""

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    @property
    def equation(self) -> str:
        return f"y = x if x > 0 else {self.alpha} * x"

    def _f(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def _f_prime(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.alpha)

class Sigmoid(Activation):
    """Sigmoid: y = 1 / (1 + exp(-x)). Backprop: dy/dx = y * (1 - y)."""

    @property
    def equation(self) -> str:
        return "y = 1 / (1 + exp(-x))"

    def _f(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _f_prime(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y * (1.0 - y)

# -----------------------------------------------------------------------------
# Loss node: takes (prediction, target), returns scalar; backward gives dL/d(pred)
# -----------------------------------------------------------------------------

class LossNode(Node, ABC):
    """Abstract loss. Forward returns scalar loss; backward returns gradient dL/d(prediction)."""

    def __init__(self):
        super().__init__()
        self._last_pred = None
        self._last_target = None

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        prediction = np.asarray(prediction, dtype=float)
        target = np.asarray(target, dtype=float)
        self._record_input(prediction)
        self._last_pred = prediction
        self._last_target = target
        return self._loss_value(prediction, target)

    def backward(self, upstream: float = 1.0) -> np.ndarray:
        """Upstream is dL/d(loss) = 1 when loss is the final scalar. Returns dL/d(prediction)."""
        return self._loss_gradient(self._last_pred, self._last_target) * upstream

    @abstractmethod
    def _loss_value(self, pred: np.ndarray, target: np.ndarray) -> float:
        pass

    @abstractmethod
    def _loss_gradient(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass


class SquaredLoss(LossNode):
    """L = sum((target - pred)^2). dL/d(pred) = -2 * (target - pred)."""

    @property
    def equation(self) -> str:
        return "L = sum((target - y)^2)"

    def _loss_value(self, pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.sum((target - pred) ** 2))

    def _loss_gradient(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return -2.0 * (target - pred)



