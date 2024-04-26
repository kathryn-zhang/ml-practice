from numpy import array
from numpy.linalg import eig

# Define the matrix A
A = array([[4, 1], [2, 3]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(A)

print(eigenvalues)
print(eigenvectors)