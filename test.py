import numpy as np

def generate_matrix(n):
    # Generate a random NxN matrix
    matrix = np.random.rand(n, n)
    return matrix

def check_inverse(matrix):
    # Find the inverse of the matrix
    inverse_matrix = np.linalg.inv(matrix)

    # Check if the product of the matrix and its inverse is the identity matrix
    identity_matrix = np.dot(matrix, inverse_matrix)

    # Check if each element in the identity matrix is very close to 1.0
    identity_check = np.allclose(identity_matrix, np.eye(matrix.shape[0]))

    return identity_matrix, identity_check

# Set the size of the matrix (N)
matrix_size = 2000  # You can change this to any positive integer

# Generate a random NxN matrix
original_matrix = generate_matrix(matrix_size)

# Check if the matrix has an inverse and if the inverse multiplied by the original gives the identity matrix
identity, result = check_inverse(original_matrix)
print("\nHas Inverse:", result)
