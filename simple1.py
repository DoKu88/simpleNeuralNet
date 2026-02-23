"""
MLP built from node classes. Each node tracks inputs, has a forward equation,
and a layer-specific backprop. Uses an abstract base class with concrete
implementations: LinearLayer, activations, and Loss.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from abc import ABC, abstractmethod

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
        self.W = rng.standard_normal((out_features, in_features)) * 0.1
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

        # dL/db = dL/dy (d(y)/db = I)
        self.dLdb = upstream.copy()

        # dL/dW: dL/dW_ij = (dL/dy_i) * x_j  -> (out, in)
        self.dLdW = np.outer(upstream, x)

        # dL/dx = W.T @ (dL/dy)
        dx = self.W.T @ upstream
        return dx

    def update_weights(self, lr: float) -> None:
        """Gradient descent step: W -= lr * dLdW, b -= lr * dLdb. Call after backward."""
        self.W -= lr * self.dLdW
        self.b -= lr * self.dLdb

class ReLU(Activation):
    """ReLU: y = max(0, x). Backprop: pass upstream where x > 0."""

    @property
    def equation(self) -> str:
        return "y = max(0, x)"

    def _f(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _f_prime(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


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

def build_dataset():
    # Option A — two interlocking half-circles (non-linear)
    X, y = make_moons(n_samples=400, noise=0.15, random_state=42)
    y = y.reshape(-1, 1)           # make it (N,1)

    # Option B — concentric circles (more non-linear)
    # X, y = make_circles(n_samples=400, noise=0.15, factor=0.4, random_state=42)
    # y = y.reshape(-1, 1)

    # Split
    np.random.seed(0)
    idx = np.random.permutation(len(X))
    train_size = 320
    X_train, y_train = X[idx[:train_size]], y[idx[:train_size]]
    X_test,  y_test  = X[idx[train_size:]], y[idx[train_size:]]

    return X_train, y_train, X_test, y_test

def plot_dataset(X_train, y_train, X_test, y_test):
    """Plots the training and test splits of the dataset."""

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), cmap='bwr', edgecolor='k', marker='o', label='Train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), cmap='cool', edgecolor='k', marker='s', alpha=0.6, label='Test')
    plt.title('Dataset: training and test splits')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Example: linear -> ReLU -> linear -> squared loss
# -----------------------------------------------------------------------------

def main():
    np.random.seed(42)

    X_train, y_train, X_test, y_test = build_dataset()
    plot_dataset(X_train, y_train, X_test, y_test)
    n_train = len(X_train)

    # Build nodes (input dim 2, output dim 1 to match dataset)
    linear1 = LinearLayer(2, 4)
    relu = ReLU()
    linear2 = LinearLayer(4, 1)
    loss_fn = SquaredLoss()

    lr = 0.1
    num_iters = 200
    losses = []

    for step in range(num_iters):
        epoch_loss = 0.0
        for i in range(n_train):
            x = X_train[i]
            y = np.ravel(y_train[i])

            # Forward
            h = linear1.forward(x)
            h = relu.forward(h)
            pred = linear2.forward(h)
            loss = loss_fn.forward(pred, y)
            epoch_loss += loss

            # Backward (upstream = 1 for loss); y = prediction in loss L(target, y)
            dL_dy = loss_fn.backward(1.0)
            dL_dh = linear2.backward(dL_dy)
            dL_dh = relu.backward(dL_dh)
            linear1.backward(dL_dh)

            # Gradient descent (only linear layers have parameters)
            linear2.update_weights(lr)
            linear1.update_weights(lr)

        losses.append(epoch_loss / n_train)

    # Final forward (average loss on training set)
    total_loss = 0.0
    for i in range(n_train):
        x = X_train[i]
        y = np.ravel(y_train[i])
        h = linear1.forward(x)
        h = relu.forward(h)
        pred = linear2.forward(h)
        total_loss += loss_fn.forward(pred, y)
    avg_loss = total_loss / n_train

    # Sample prediction for display
    x0, y0 = X_train[0], np.ravel(y_train[0])
    h = linear1.forward(x0)
    h = relu.forward(h)
    pred = linear2.forward(h)

    print("=== After training (simple1 node-based MLP) ===")
    print("Equations:")
    print(f"  Linear1: {linear1.equation}")
    print(f"  ReLU:    {relu.equation}")
    print(f"  Linear2: {linear2.equation}")
    print(f"  Loss:    {loss_fn.equation}")
    print(f"sample predicted: {pred}")
    print(f"sample ground truth: {y0}")
    print(f"mean train loss: {avg_loss:.6f}")
    print(f"inputs tracked by linear1: {len(linear1._inputs)} (one per forward)")

    # Plot loss per iteration
    plt.figure(figsize=(8, 5))
    plt.plot(range(num_iters), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs iteration")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
