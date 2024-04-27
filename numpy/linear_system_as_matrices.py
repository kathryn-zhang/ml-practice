import numpy as np
import matplotlib.pyplot as plt

# from utils import plot_lines

# System of linear Equations as Matrices
A = np.array([
    [-1, 3],
    [3, 2]
], dtype=np.dtype(float))

b = np.array([7, 1], dtype=np.dtype(float))

print("Matrix A:")
print(A)
# Matrix A:
# [[-1.  3.]
#  [ 3.  2.]]

print("\nArray b:")
print(b)
# Array b:
# [7. 1.]

print(f"Shape of A: {A.shape}")  # Shape of A: (2, 2)
print(f"Shape of b: {b.shape}")  # Shape of b: (2,)

# print(f"Shape of A: {np.shape(A)}")
# print(f"Shape of A: {np.shape(b)}")

# Solve linear equations
x = np.linalg.solve(A, b)

print(f"Solution: {x}")  # Solution: [-1.  2.]

# Evaluating Determinant of a Matrix
d = np.linalg.det(A)

print(f"Determinant of matrix A: {d:.2f}")
# Determinant of matrix A: -11.00

#  Visualizing 2x2 Systems as Plotlines
A_system = np.hstack((A, b.reshape((2, 1))))

print(A_system)
# [[-1.  3.  7.]
#  [ 3.  2.  1.]]

print(A_system[1])  # [3. 2. 1.]

# Solve the system of equations
solution = np.linalg.solve(A, b)

# Graphical Representation of the Solution
# Define the range for x1
x1 = np.linspace(-10, 10, 400)

# Calculate x2 from the rearranged equations
x2_first_eq = (7 + x1) / 3
x2_second_eq = (1 - 3 * x1) / 2

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x1, x2_first_eq, label='x2 = (7 + x1) / 3')
plt.plot(x1, x2_second_eq, label='x2 = (1 - 3 * x1) / 2')

# Mark the solution on the plot
plt.scatter(*solution, color='red', zorder=5)
plt.annotate(f'Solution ({solution[0]:.2f}, {solution[1]:.2f})',
             (solution[0], solution[1]), textcoords="offset points",
             xytext=(10,-10), ha='center', color='red')

# Add labels and legend
plt.title('Graphical Representation of the System of Equations')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True)
plt.legend()

# Show the plot
#plt.show()

#  System of Linear Equations with No Solutions
A_2 = np.array([
        [-1, 3],
        [3, -9]
    ], dtype=np.dtype(float))

b_2 = np.array([7, 1], dtype=np.dtype(float))

d_2 = np.linalg.det(A_2)

print(f"Determinant of matrix A_2: {d_2:.2f}")
#  Determinant of matrix A_2: -0.00, no unique solution

try:
    x_2 = np.linalg.solve(A_2, b_2)
    print("Solution:", x_2)  # BUG: Solution: [1.32105589e+17 4.40351964e+16]
except np.linalg.LinAlgError as err:
    print("Error encountered:", err)

# System of Linear Equations with an Infinite Number of Solutions
b_3 = np.array([7, -21], dtype=np.dtype(float))

A_3_system = np.hstack((A_2, b_3.reshape((2, 1))))
print(A_3_system)
# [[ -1.   3.   7.]
#  [  3.  -9. -21.]]