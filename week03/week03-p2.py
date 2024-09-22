import numpy as np
import time

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

# Generate a non-PSD matrix
def generate_non_psd_matrix(n):
    sigma = np.full((n, n), 0.9)
    np.fill_diagonal(sigma, 1.0)
    sigma[0, 1] = sigma[1, 0] = 0.7357
    return sigma

# Calculate Frobenius norm
def frobenius_norm(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2)

# Test methods for different N values
def test_methods_for_different_N():
    Ns = [100, 200, 300, 400, 500]  # Varying matrix sizes
    results = []
    
    for N in Ns:
        print(f"\nTesting for N={N}...")

        non_psd_matrix = generate_non_psd_matrix(N)
        
        # Higham PSD
        start_time = time.time()
        higham_psd_result = higham_psd(non_psd_matrix)
        higham_psd_time = time.time() - start_time
        
        # Near PSD 
        start_time = time.time()
        near_psd_result = near_psd(non_psd_matrix)
        near_psd_time = time.time() - start_time
        
        # Frobenius norm comparison
        frobenius_higham = frobenius_norm(non_psd_matrix, higham_psd_result)
        frobenius_near = frobenius_norm(non_psd_matrix, near_psd_result)
        
        results.append({
            "N": N,
            "Frobenius Norm (Higham PSD)": frobenius_higham,
            "Frobenius Norm (Near PSD)": frobenius_near,
            "Time (Higham PSD)": higham_psd_time,
            "Time (Near PSD)": near_psd_time
        })
    
    return results

results = test_methods_for_different_N()

for result in results:
    print(f"\nN = {result['N']}")
    print(f"Higham PSD: Frobenius Norm = {result['Frobenius Norm (Higham PSD)']:.4f}, Time = {result['Time (Higham PSD)']:.4f} seconds")
    print(f"Near PSD: Frobenius Norm = {result['Frobenius Norm (Near PSD)']:.4f}, Time = {result['Time (Near PSD)']:.4f} seconds")
