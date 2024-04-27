import numpy as np

A = np.array([
    [1, 2, 1],
    [2, 1, 1],
    [-1, 2, 1]
], dtype=np.dtype(float))

# Evaluating Determinant of a Matrix
d = np.linalg.det(A)
print(d)

B = np.array([
    [-3, 8, 1],
    [2, 2, -1],
    [-5, 6, 2]
], dtype=np.dtype(float))

# Evaluating Determinant of a Matrix
d = np.linalg.det(B)
print(d)