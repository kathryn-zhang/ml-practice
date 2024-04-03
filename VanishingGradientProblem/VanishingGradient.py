import numpy as np


# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Simulate gradients in a deep network
layer_inputs = np.random.normal(size=10)  # Simulated input for 10 layers
gradients = np.ones(10)  # Initial gradients (dummy values)

for i in range(len(layer_inputs)):
    gradients[i] = gradients[i - 1] * sigmoid_derivative(layer_inputs[i])

#  The output will show decreasing gradient magnitudes, demonstrating the vanishing gradient problem.
print(gradients)
