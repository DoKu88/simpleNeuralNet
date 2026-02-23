import numpy as np
import matplotlib.pyplot as plt

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
    # In numpy it's (rows, columns)
    x = np.asarray([0.5, 0.5, 0.5])
    m = np.random.rand(2, 3)
    bias = np.random.rand(2)
    ground_truth = np.asarray([0.75, 0.10])

    # Forward
    y, diff, loss = forward_pass(x, m, bias, ground_truth)
    print("===Forward Pass===")
    print(f"predicted: {y}")
    print(f"ground truth: {ground_truth}")
    print(f"difference {diff}")
    print(f"loss: {loss}")
    print("\n")

    # Backward
    dLdy, dydm, dydb, dLdm, dLdb = backward_pass(ground_truth, y, x, m, bias)
    print("===Gradients===")
    print(f"dL/dy: {dLdy}, shape: {dLdy.shape}")
    print(f"dy/dm: {dydm}, shape: {dydm.shape}")
    print(f"dy/db: {dydb}, shape: {dydb.shape}")
    print(f"dL/db: {dLdb}, shape: {dLdb.shape}")
    print(f"dL/dm: {dLdm}, shape: {dLdm.shape}")


if __name__ == "__main__":
    main()
