import numpy as np

# Generate sample data
np.random.seed(42)
x = np.random.randn(100, 1)
y = 2 * x + 3 + np.random.randn(100, 1) * 0.5

# Initialize parameters
w = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01
num_epochs = 1000

# Stochastic Gradient Descent
for epoch in range(num_epochs):
    # Shuffle the data
    indices = np.random.permutation(len(x))
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # Iterate over the shuffled data
    for xi, yi in zip(x_shuffled, y_shuffled):
        # Calculate gradients for a single sample
        yhat = w * xi + b
        dldw = -2 * (yi - yhat) * xi
        dldb = -2 * (yi - yhat)

        # Update the parameters
        w = w - learning_rate * dldw
        b = b - learning_rate * dldb

    # Calculate and print the loss
    yhat = w * x + b
    loss = np.mean((y - yhat) ** 2)
    print(f"Epoch {epoch+1}, Loss: {loss}, w: {w}, b: {b}")

# Print the final model
print(f"Final model: y = {w}x + {b}")