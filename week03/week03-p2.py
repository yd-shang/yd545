import numpy as np
import time

def chol_psd(root, a):
    n = a.shape[0]
    root[:] = 0.0

    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        temp = a[j, j] - s
        
        if temp < 0:
            temp = 0.0
        
        root[j, j] = np.sqrt(temp)
        
        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir

# Higham's nearest PSD method
def higham_psd(matrix, max_iter=100):
    Y = matrix.copy()
    for _ in range(max_iter):
        R = Y - np.diag(np.diag(Y))
        X = np.clip(R, 0, None)
        Y = X + np.diag(np.diag(matrix))
        eigenvalues = np.linalg.eigvals(Y)
        if np.all(eigenvalues >= 0):
            break
    return Y

# Approximate PSD adjustment
def near_psd(a, epsilon=0.0):
    n = a.shape[0]
    out = np.copy(a)
    if np.count_nonzero(np.diag(out) == 1.0) != n:
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    eigenvalues, eigenvectors = np.linalg.eigh(out)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    T = 1.0 / (eigenvectors * eigenvectors @ eigenvalues)
    T = np.diag(np.sqrt(T))
    L = np.diag(np.sqrt(eigenvalues))
    B = T @ eigenvectors @ L
    out = B @ B.T
    if np.count_nonzero(np.diag(out) == 1.0) != n:
        invSD = np.diag(1.0 / np.diag(out))
        out = invSD @ out @ invSD
    return out

# Generate a 500x500 non-PSD matrix
def generate_non_psd_matrix(n=500):
    sigma = np.full((n, n), 0.9)
    np.fill_diagonal(sigma, 1.0)
    sigma[0, 1] = sigma[1, 0] = 0.7357
    return sigma

# Calculate Frobenius norm
def frobenius_norm(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2)


def test_methods():
    non_psd_matrix = generate_non_psd_matrix(500)
    
    # Cholesky PSD 
    root = np.zeros((500, 500))
    start_time = time.time()
    chol_psd(root, non_psd_matrix)
    chol_psd_result = root @ root.T
    chol_psd_time = time.time() - start_time
    
    # Higham PSD
    start_time = time.time()
    higham_psd_result = higham_psd(non_psd_matrix)
    higham_psd_time = time.time() - start_time
    
    # Near PSD 
    start_time = time.time()
    near_psd_result = near_psd(non_psd_matrix)
    near_psd_time = time.time() - start_time
    
    # Frobenius norm comparison
    frobenius_chol = frobenius_norm(non_psd_matrix, chol_psd_result)
    frobenius_higham = frobenius_norm(non_psd_matrix, higham_psd_result)
    frobenius_near = frobenius_norm(non_psd_matrix, near_psd_result)

    print("Runtime (Cholesky PSD): {:.4f} seconds".format(chol_psd_time))
    print("Runtime (Higham PSD): {:.4f} seconds".format(higham_psd_time))
    print("Runtime (Near PSD): {:.4f} seconds".format(near_psd_time))
    print("Frobenius Norm (Cholesky PSD): {:.4f}".format(frobenius_chol))
    print("Frobenius Norm (Higham PSD): {:.4f}".format(frobenius_higham))
    print("Frobenius Norm (Near PSD): {:.4f}".format(frobenius_near))
    
    return {
        "Frobenius Norm (Cholesky PSD)": frobenius_chol,
        "Frobenius Norm (Higham PSD)": frobenius_higham,
        "Frobenius Norm (Near PSD)": frobenius_near,
        "Time (Cholesky PSD)": chol_psd_time,
        "Time (Higham PSD)": higham_psd_time,
        "Time (Near PSD)": near_psd_time
    }

results = test_methods()
