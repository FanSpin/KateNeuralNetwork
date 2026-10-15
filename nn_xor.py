import numpy as np

# --- Data: XOR truth table ---
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# --- Hyperparameters ---
lr = 0.5          # learning rate
epochs = 10000    # training iterations

# --- Initialize weights (random, small) ---
np.random.seed(42)
W1 = np.random.randn(2, 4)   # 2 inputs → 4 hidden neurons
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)   # 4 hidden → 1 output
b2 = np.zeros((1, 1))

# --- Activation function ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Training loop ---
for i in range(epochs):
    # Forward pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # Backward pass (gradients)
    dz2 = a2 - y                          # output error
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * a1 * (1 - a1)   # hidden error
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update weights
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # Print progress every 2000 epochs
    if i % 2000 == 0:
        print(f"Epoch {i:5d}  loss = {np.mean((a2 - y)**2):.6f}")

# --- Test: predict XOR ---
print("\nInput  →  Output")
for row, target in zip(X, y):
    z1 = row @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    pred = sigmoid(z2)
    print(f"  {row} → {pred[0][0]:.4f}  (target: {int(target[0])})")
