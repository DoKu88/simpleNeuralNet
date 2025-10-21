import numpy as np 
from abc import ABC 

def sigmoid(z):
    sig = 1 / (1 + np.exp(-1 * z))
    return np.asarray([sig])

def loss(pred, y):
    result = 0
    for pred_i, y_i in zip(pred, y):
        result += (pred_i - y_i)**2 
    
    result = result / len(pred)
    return result 

def main():

    input0 = np.asarray([1.0, 2.0, 3.0])
    output = np.asarray([0.5])

    weights = np.random.rand(3,)
    bias = 1

    weights1 = np.concatenate([weights, np.asarray([bias])])
    input1 = np.concatenate([input0, np.asarray([1])])

    z = input1 @ weights1
    z_sig = sigmoid(z)

    loss_1 = loss(z_sig, output)

    print(f"input1 {input1}")
    print(f"weights1 {weights1}")
    print(f"z {z}")
    print(f"z_sig {z_sig}")
    print(f"loss {loss_1}")

    return

if __name__ == "__main__":
    print("Hello world!")
    main()