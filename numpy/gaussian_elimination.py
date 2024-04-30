import numpy as np

def swap_rows(M, row_index_1, row_index_2):
    """
    Swap rows in the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to perform row swaps on.
    - row_index_1 (int): Index of the first row to be swapped.
    - row_index_2 (int): Index of the second row to be swapped.
    """

    # Copy matrix M so the changes do not affect the original matrix.
    M = M.copy()
    # Swap indexes
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M


def get_index_first_non_zero_value_from_column(M, column, starting_row):
    """
    Retrieve the index of the first non-zero value in a specified column of the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to search for non-zero values.
    - column (int): The index of the column to search.
    - starting_row (int): The starting row index for the search.

    Returns:
    int: The index of the first non-zero value in the specified column, starting from the given row.
                Returns -1 if no non-zero value is found.
    """
    # Get the column array starting from the specified row
    column_array = M[starting_row:,column]
    print(f'column array: {column_array}')
    for i, val in enumerate(column_array):
        # Iterate over every value in the column array.
        # To check for non-zero values, you must always use np.isclose instead of doing "val == 0".
        if not np.isclose(val, 0, atol = 1e-5):
            # If one non zero value is found, then adjust the index to match the correct index in the matrix and return it.
            index = i + starting_row
            return index
    # If no non-zero value is found below it, return -1.
    return -1

N = np.array([
[0, 5, -3 ,6 ,8],
[0, 6, 3, 8, 1],
[0, 0, 0, 0, 0],
[0, 0, 0 ,0 ,7],
[0, 2, 1, 0, 4]
]
)
print(N)

print(get_index_first_non_zero_value_from_column(N, column = 0, starting_row = 0))
print(get_index_first_non_zero_value_from_column(N, column = -1, starting_row = 2))

def get_index_first_non_zero_value_from_row(M, row, augmented=False):
    """
    Find the index of the first non-zero value in the specified row of the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to search for non-zero values.
    - row (int): The index of the row to search.
    - augmented (bool): Pass this True if you are dealing with an augmented matrix,
                        so it will ignore the constant values (the last column in the augmented matrix).

    Returns:
    int: The index of the first non-zero value in the specified row.
                Returns -1 if no non-zero value is found.
    """

    # Create a copy to avoid modifying the original matrix
    M = M.copy()

    # If it is an augmented matrix, then ignore the constant values
    if augmented == True:
        # Isolating the coefficient matrix (removing the constant terms)
        M = M[:, :-1]
        print(f'M: {M}')

    # Get the desired row
    row_array = M[row]
    for i, val in enumerate(row_array):
        # If finds a non zero value, returns the index. Otherwise returns -1.
        if not np.isclose(val, 0, atol=1e-5):
            return i
    return -1


print(f'Output for row 2: {get_index_first_non_zero_value_from_row(N, 2)}')
print(f'Output for row 3: {get_index_first_non_zero_value_from_row(N, 3)}')

def augmented_matrix(A, B):
    """
    Create an augmented matrix by horizontally stacking two matrices A and B.

    Parameters:
    - A (numpy.array): First matrix.
    - B (numpy.array): Second matrix.

    Returns:
    - numpy.array: Augmented matrix obtained by horizontally stacking A and B.
    """
    augmented_M = np.hstack((A,B))
    return augmented_M


A = np.array([[1,2,3], [3,4,5], [4,5,6]])
B = np.array([[1], [5], [7]])

print(augmented_matrix(A,B))


def row_echelon_form(A, B):
    """
    Utilizes elementary row operations to transform a given set of matrices,
    which represent the coefficients and constant terms of a linear system, into row echelon form.

    Parameters:
    - A (numpy.array): The input square matrix of coefficients.
    - B (numpy.array): The input column matrix of constant terms

    Returns:
    numpy.array: A new augmented matrix in row echelon form with pivots as 1.
    """

    # Calculate the determinant of matrix A
    det_A = np.linalg.det(A)

    # Return "Singular system" if determinant is close to zero
    if np.isclose(det_A, 0, atol=1e-5):
        return 'Singular system'

    # Create copies and ensure data type is float for accurate division
    A = A.astype('float64')
    B = B.astype('float64')

    # Create the augmented matrix from A and B
    M = np.hstack((A, B))

    # Number of rows and columns in matrix A
    num_rows, num_cols = A.shape

    # Start row reduction
    for i in range(num_rows):
        # Find the pivot element
        if np.isclose(M[i, i], 0):
            # Find non-zero pivot below current row and swap if needed
            for k in range(i + 1, num_rows):
                if not np.isclose(M[k, i], 0):
                    M[[i, k]] = M[[k, i]]  # Swap rows
                    break

        # Set the pivot element to 1
        pivot = M[i, i]
        if not np.isclose(pivot, 0):
            M[i] = M[i] / pivot  # Normalize the pivot row

        # Eliminate all elements below the pivot
        for j in range(i + 1, num_rows):
            factor = M[j, i]
            M[j] -= factor * M[i]

    return M

A = np.array([[1,2,3],[0,1,0], [0,0,5]])
B = np.array([[1], [2], [4]])
print(row_echelon_form(A,B))


def back_substitution(M):
    """
    Perform back substitution on an augmented matrix (with unique solution) in reduced row echelon form to find the solution to the linear system.

    Parameters:
    - M (numpy.array): The augmented matrix in row echelon form with unitary pivots (n x n+1).

    Returns:
    numpy.array: The solution vector of the linear system.
    """

    # Make a copy of the input matrix to avoid modifying the original
    M = M.copy()

    # Get the number of rows (and columns) in the matrix of coefficients
    num_rows = M.shape[0]

    # Initialize the solution array with zeros
    solution = np.zeros(num_rows)

    # Iterate from the last row to the first row
    for i in reversed(range(num_rows)):
        # Start with the right-hand side of the equation
        sum_ax = M[i, -1]

        # Subtract the sum of the coefficients multiplied by the known solutions
        for j in range(i + 1, num_rows):
            sum_ax -= M[i, j] * solution[j]

        # The pivot element, which should be 1 for reduced row echelon form
        pivot = M[i, i]
        if pivot == 0:
            raise ValueError("Matrix is singular and cannot be solved by back substitution.")

        # Calculate the solution for the current variable
        solution[i] = sum_ax / pivot

    return solution


def gaussian_elimination(A, B):
    """
    Solve a linear system represented by an augmented matrix using the Gaussian elimination method.

    Parameters:
    - A (numpy.array): Square matrix of size n x n representing the coefficients of the linear system
    - B (numpy.array): Column matrix of size 1 x n representing the constant terms.

    Returns:
    numpy.array or str: The solution vector if a unique solution exists, or a string indicating the type of solution.
    """

    ### START CODE HERE ###

    # Get the matrix in row echelon form
    row_echelon_M = row_echelon_form(A, B)

    # If the system is non-singular, then perform back substitution to get the result.
    # Since the function row_echelon_form returns a string if there is no solution, let's check for that.
    # The function isinstance checks if the first argument has the type as the second argument, returning True if it does and False otherwise.
    if not isinstance(row_echelon_M, str):
        solution = back_substitution(row_echelon_M)

    ### END SOLUTION HERE ###

    return solution