import numpy as np

#  Solving Systems of Linear Equations using Matrices
A = np.array([
        [4, -3, 1],
        [2, 1, 3],
        [-1, 2, -5]
    ], dtype=np.dtype(float))

b = np.array([-10, 0, 17], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)

# Matrix A:
# [[ 4. -3.  1.]
#  [ 2.  1.  3.]
#  [-1.  2. -5.]]
#
# Array b:
# [-10.   0.  17.]

print(f"Shape of A: {np.shape(A)}")
print(f"Shape of b: {np.shape(b)}")
# Shape of A: (3, 3)
# Shape of b: (3,)

x = np.linalg.solve(A, b)

print(f"Solution: {x}")

# Evaluating the determinant of a matrix
d = np.linalg.det(A)

print(f"Determinant of matrix A: {d:.2f}")  # -60.00

