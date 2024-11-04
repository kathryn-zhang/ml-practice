import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
x = np.linspace(-10, 10, 400)

# Define the functions
y1 = x**2
y2 = 2*x

# Plot the functions
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label="y = x^2")
plt.plot(x, y2, label="y = 2x")

# Add titles and labels
plt.title("Plot of y = x^2 and y = 2x")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Define the range of x values for y=log(x) and y=1/x
x_log = np.linspace(0.1, 10, 400)  # Avoid zero for log(x) to prevent undefined values
x_inv = np.linspace(0.1, 10, 400)  # Avoid zero for 1/x to prevent division by zero

# Define the functions
y_log = np.log(x_log)
y_inv = 1 / x_inv

# Plot the functions
plt.figure(figsize=(10, 6))
plt.plot(x_log, y_log, label="y = log(x)")
plt.plot(x_inv, y_inv, label="y = 1/x")

# Add titles and labels
plt.title("Plot of y = log(x) and y = 1/x")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

# Add a legend
plt.legend()

# Show the plot
plt.show()