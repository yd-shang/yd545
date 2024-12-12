import numpy as np

# Check if the covariance matrix is positive semi-definite (PSD)
def is_psd(matrix):
    # A matrix is PSD if all its eigenvalues are non-negative
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues >= 0)

# Fix a non-PSD matrix using the "near_psd" method
def near_psd(matrix, epsilon=1e-8):
    """Make a matrix near positive semi-definite by fixing eigenvalues."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Replace negative eigenvalues with a small positive value
    fixed_eigenvalues = np.maximum(eigenvalues, epsilon)
    # Reconstruct the matrix
    fixed_matrix = eigenvectors @ np.diag(fixed_eigenvalues) @ eigenvectors.T
    # Ensure symmetry
    fixed_matrix = (fixed_matrix + fixed_matrix.T) / 2
    return fixed_matrix


