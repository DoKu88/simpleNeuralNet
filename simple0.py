import numpy as np

# let's make a simple network: 
# y = mx + b 

# In numpy it's (rows, columns)

x = np.asarray([0.5, 0.5, 0.5])

m = np.random.rand(2, 3)
bias = np.random.rand(2)

y = m @ x + bias 

ground_truth = np.asarray([0.75, 0.10])
diff = ground_truth - y

# squared loss: (ground_truth - y)**2 
loss = np.cumsum((diff)**2)[-1]

print(f"predicted: {y}")
print(f"ground truth: {ground_truth}")
print(f"difference {diff}")
print(f"loss: {loss}")

# write gradients to update parameters with 
dLdy = -2 * (ground_truth - y)
# broadcast x to same shape as m (each row = x)
dydm = np.broadcast_to(x[np.newaxis, :], m.shape).copy()
dydb = np.identity(bias.shape[0])

print("Gradients:")
print(f"dL/dy: {dLdy}, shape: {dLdy.shape}")
print(f"dy/dm: {dydm}, shape: {dydm.shape}")
print(f"dy/db: {dydb}, shape: {dydb.shape}")

# dL/db = dL/dy @ dy/db  (dLdy (2,) @ (2,2) -> (2,))
dLdb = dLdy @ dydb

# dL/dm: chain rule is dL/dm_ij = (dL/dy_i) * (dy_i/dm_ij) = (dL/dy_i) * x_j
# So we need (2,3) where [i,j] = dLdy[i] * dydm[i,j]. Element-wise, not @
dLdm = dLdy[:, np.newaxis] * dydm

print(f"dL/db: {dLdb}, shape: {dLdb.shape}")
print(f"dL/dm: {dLdm}, shape: {dLdm.shape}")