import numpy as np
import matplotlib.pyplot as plt

# Step 2: Define sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Step 3: Generate input range
x = np.linspace(-10, 10, 100)

# Step 4: Compute outputs
y = sigmoid(x)

# Step 5: Plot
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.title("Sigmoid Activation Function")
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
plt.show()
