import numpy as np

# Matrix Multiplication
A = np.array([[4, 9, 9], [9, 1, 6], [9, 2, 3]])
print("Matrix A (3 by 3):\n", A)

B = np.array([[2, 2], [5, 7], [4, 4]])
print("Matrix B (3 by 2):\n", B)

# multiply matrices  𝐴  and  𝐵  using NumPy package function np.matmul()
print(np.matmul(A, B))
# [[ 89 107]
#  [ 47  49]
#  [ 40  44]]

# Python operator @ will also work here giving the same result
print(A @ B)
# [[ 89 107]
#  [ 47  49]
#  [ 40  44]]

# Also works. NumPy broadcasts this dot product operation to all rows and all columns
print(np.dot(A, B))
# [[ 89 107]
#  [ 47  49]
#  [ 40  44]]

# Matrix Convention and Broadcasting
# BA will not work!
try:
    np.matmul(B, A)
except ValueError as err:
    print(err)

x = np.array([1, -2, -5])
y = np.array([4, 3, -1])

print("Shape of vector x:", x.shape)
print("Number of dimensions of vector x:", x.ndim)
print("Shape of vector x, reshaped to a matrix:", x.reshape((3, 1)).shape)
print("Number of dimensions of vector x, reshaped to a matrix:", x.reshape((3, 1)).ndim)

# Shape of vector x: (3,)
# Number of dimensions of vector x: 1
# Shape of vector x, reshaped to a matrix: (3, 1)
# Number of dimensions of vector x, reshaped to a matrix: 2

print(x.reshape((3, 1)))
# [[ 1]
#  [-2]
#  [-5]]
np.matmul(x,y)  # 3

