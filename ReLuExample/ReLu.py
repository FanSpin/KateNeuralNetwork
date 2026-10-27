import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

# Random init
w1 = np.random.randn(2, 2)   # input → hidden
w2 = np.random.randn(2, 1)   # hidden → output
lr = 0.01

for t in range(500):
    h = x @ w1              # ← matrix multiplication
    h_relu = np.maximum(h, 0)   # ← ReLU applied here
    y_pred = h_relu @ w2

    # Backprop
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T @ grad_y_pred
    grad_h_relu = grad_y_pred @ w2.T
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0           # derivative of ReLU: 0 where h<0
    grad_w1 = x.T @ grad_h

    w1 -= lr * grad_w1
    w2 -= lr * grad_w2

print(np.round(y_pred))  # [[0],[1],[1],[0]]
