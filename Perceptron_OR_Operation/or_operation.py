def perceptron_train(inputs, outputs, learning_rate, epochs):
    # Initialize weights and bias
    weights = [0, 0]  # w1, w2
    bias = 0

    for _ in range(epochs):
        for inp, desired_output in zip(inputs, outputs):
            # Calculate the actual output
            weighted_sum = sum(w * i for w, i in zip(weights, inp)) + bias
            actual_output = 1 if weighted_sum > 0 else 0

            # Update weights and bias if there is a misclassification
            if actual_output != desired_output:
                for j in range(len(weights)):
                    weights[j] += learning_rate * (desired_output - actual_output) * inp[j]
                bias += learning_rate * (desired_output - actual_output)
    
    return weights, bias

# OR operation data
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
outputs = [0, 1, 1, 1]

# Training parameters
learning_rate = 0.1
epochs = 10  # Number of times to iterate through the entire dataset

# Train the perceptron
final_weights, final_bias = perceptron_train(inputs, outputs, learning_rate, epochs)
final_weights, final_bias


