import numpy as np

# Hypothetical dataset
Area = np.array([50, 60, 70, 80, 90])  # Areas in square meters
Price = np.array([150, 180, 200, 220, 240])  # Prices in thousand dollars

# Normalize the Area and Price
Area_norm = (Area - np.mean(Area)) / np.std(Area)
Price_norm = (Price - np.mean(Price)) / np.std(Price)

# Initialize parameters
theta_0 = 0
theta_1 = 0
alpha = 0.01  # Learning rate
iterations = 100  # Number of iterations for gradient descent
n = len(Area)  # Number of data points

# Gradient Descent with normalized data
for _ in range(iterations):
    # Predicted price with current parameters (normalized)
    predicted_price_norm = theta_0 + theta_1 * Area_norm

    # Partial derivatives of the cost function with respect to theta_0 and theta_1
    d_theta_0 = (1 / n) * sum(predicted_price_norm - Price_norm)
    d_theta_1 = (1 / n) * sum((predicted_price_norm - Price_norm) * Area_norm)

    # Update the parameters
    theta_0 = theta_0 - alpha * d_theta_0
    theta_1 = theta_1 - alpha * d_theta_1

# Final parameters
print("Theta 0 (Intercept):", theta_0)
print("Theta 1 (Slope):", theta_1)