import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import petname

# Run identifier for saved plots: [timestamp]_[RUN_NAME]_[plot_type].png
RUN_NAME = petname.Generate(2, "_")
OUTPUT_DIR = "outputs"

# -----------------------------------------------------------------------------
# Forward pass: y = m @ x + b, then loss
# -----------------------------------------------------------------------------

def forward_pass(x, m, bias, ground_truth):
    """y = m @ x + bias; compute diff and squared loss."""
    y = m @ x + bias
    diff = ground_truth - y
    # squared loss: (ground_truth - y)**2
    loss = np.cumsum((diff) ** 2)[-1]
    return y, diff, loss


# -----------------------------------------------------------------------------
# Backward pass: gradients for m and bias
# -----------------------------------------------------------------------------

def backward_pass(ground_truth, y, x, m, bias):
    """Compute dL/dm and dL/db from loss gradient and forward quantities."""
    dLdy = -2 * (ground_truth - y)
    # broadcast x to same shape as m (each row = x)
    dydm = np.broadcast_to(x[np.newaxis, :], m.shape).copy()
    dydb = np.identity(bias.shape[0])

    # dL/db = dL/dy @ dy/db  (dLdy (2,) @ (2,2) -> (2,))
    dLdb = dLdy @ dydb

    # dL/dm: chain rule is dL/dm_ij = (dL/dy_i) * (dy_i/dm_ij) = (dL/dy_i) * x_j
    # So we need (2,3) where [i,j] = dLdy[i] * dydm[i,j]. Element-wise, not @
    dLdm = dLdy[:, np.newaxis] * dydm

    return dLdy, dydm, dydb, dLdm, dLdb


def main():
    # Simple network: y = m @ x + b
    x = np.asarray([0.5, 0.5, 0.5])
    m = np.random.rand(2, 3)
    bias = np.random.rand(2)
    ground_truth = np.asarray([0.75, 0.10])

    num_iters = 100
    learning_rate = 0.1
    losses = []

    # One timestamp and output dir per run; plots saved as [timestamp]_[RUN_NAME]_[type].png
    run_ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def plot_path(plot_type: str) -> str:
        return os.path.join(OUTPUT_DIR, f"{run_ts}_{RUN_NAME}_{plot_type}.png")

    for i in range(num_iters):
        # Forward
        y, diff, loss = forward_pass(x, m, bias, ground_truth)
        losses.append(loss)

        # Backward
        dLdy, dydm, dydb, dLdm, dLdb = backward_pass(ground_truth, y, x, m, bias)

        # Gradient descent update
        m = m - learning_rate * dLdm
        bias = bias - learning_rate * dLdb

    # Final forward pass for printing
    y, diff, loss = forward_pass(x, m, bias, ground_truth)
    print("===After training===")
    print(f"predicted: {y}")
    print(f"ground truth: {ground_truth}")
    print(f"difference {diff}")
    print(f"final loss: {loss}")

    # Plot 1: Loss vs iteration (saved)
    plt.figure(figsize=(8, 5))
    plt.plot(range(num_iters), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs iteration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path("loss"), dpi=150, bbox_inches="tight")
    plt.show()

    # Plot 2: Predicted vs ground truth (mirrors neural_net "predictions" style)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    out_dim = len(ground_truth)
    x_pos = np.arange(out_dim)
    width = 0.35
    ax1.bar(x_pos - width / 2, ground_truth, width, label="Ground truth", color="steelblue")
    ax1.bar(x_pos + width / 2, y, width, label="Predicted", color="coral", alpha=0.9)
    ax1.set_xlabel("Output index")
    ax1.set_ylabel("Value")
    ax1.set_title("Ground truth vs predicted")
    ax1.set_xticks(x_pos)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x_pos, np.ravel(diff), color="gray", alpha=0.8)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_xlabel("Output index")
    ax2.set_ylabel("target - prediction")
    ax2.set_title("Residuals\nTarget - Predicted")
    ax2.set_xticks(x_pos)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plot_path("predictions"), dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
