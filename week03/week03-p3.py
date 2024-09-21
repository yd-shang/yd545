import numpy as np
import pandas as pd
import time
from scipy.stats import multivariate_normal

file_path = 'DailyReturn.csv'
daily_return_data = pd.read_csv(file_path)

pearson_corr = daily_return_data.corr()

var_std = daily_return_data.var()

def exponential_weighted_var(returns_data, lambda_factor=0.97):
    returns = returns_data.values
    num_assets = returns.shape[1]
    
    ew_var = np.zeros(num_assets)
    mean_returns = np.zeros(num_assets)
    
    for i in range(returns.shape[0]):
        deviation = returns[i] - mean_returns
        ew_var = lambda_factor * ew_var + (1 - lambda_factor) * deviation ** 2
        mean_returns = lambda_factor * mean_returns + (1 - lambda_factor) * returns[i]
    
    return ew_var

ew_var_std = exponential_weighted_var(daily_return_data)

def build_covariance_matrix(corr_matrix, var_vector):
    std_dev = np.sqrt(var_vector)
    cov_matrix = corr_matrix * np.outer(std_dev, std_dev)
    return cov_matrix

cov_pearson_std = build_covariance_matrix(pearson_corr, var_std)
cov_pearson_ew_var = build_covariance_matrix(pearson_corr, ew_var_std)
cov_exp_corr_std = build_covariance_matrix(pearson_corr, var_std)
cov_exp_corr_ew_var = build_covariance_matrix(pearson_corr, ew_var_std)

cov_matrices = {
    "Pearson + Std Var": cov_pearson_std,
    "Pearson + EW Var": cov_pearson_ew_var,
    "Exp Corr + Std Var": cov_exp_corr_std,
    "Exp Corr + EW Var": cov_exp_corr_ew_var
}

def simulate_direct(cov_matrix, num_simulations=25000):
    mean = np.zeros(cov_matrix.shape[0])
    simulated_data = multivariate_normal.rvs(mean=mean, cov=cov_matrix, size=num_simulations)
    return simulated_data

def simulate_pca(cov_matrix, explained_variance_ratio=1.0, num_simulations=25000):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    total_variance = np.sum(eigenvalues)
    
    variance_explained = 0
    num_components = 0
    for i in range(len(eigenvalues)-1, -1, -1):
        variance_explained += eigenvalues[i]
        num_components += 1
        if variance_explained / total_variance >= explained_variance_ratio:
            break

    reduced_eigenvalues = eigenvalues[-num_components:]
    reduced_eigenvectors = eigenvectors[:, -num_components:]
    
    simulated_data = np.dot(np.random.randn(num_simulations, num_components), np.sqrt(np.diag(reduced_eigenvalues))) @ reduced_eigenvectors.T
    return simulated_data

def calculate_frobenius_norm(input_cov, simulated_data):
    simulated_cov = np.cov(simulated_data, rowvar=False)
    frobenius_norm = np.linalg.norm(input_cov - simulated_cov)
    return frobenius_norm

for label, cov_matrix in cov_matrices.items():
    print(f"\nComparison for: {label}")
    
    # Direct Simulation
    start_time = time.time()
    direct_simulation = simulate_direct(cov_matrix)
    direct_time = time.time() - start_time
    frobenius_direct = calculate_frobenius_norm(cov_matrix, direct_simulation)
    print(f"Direct Simulation - Frobenius Norm: {frobenius_direct:.6f}, Time: {direct_time:.4f} seconds")
    
    # PCA 100% Simulation
    start_time = time.time()
    pca_100 = simulate_pca(cov_matrix, explained_variance_ratio=1.0)
    pca_100_time = time.time() - start_time
    frobenius_pca_100 = calculate_frobenius_norm(cov_matrix, pca_100)
    print(f"PCA 100% - Frobenius Norm: {frobenius_pca_100:.6f}, Time: {pca_100_time:.4f} seconds")
    
    # PCA 75% Simulation
    start_time = time.time()
    pca_75 = simulate_pca(cov_matrix, explained_variance_ratio=0.75)
    pca_75_time = time.time() - start_time
    frobenius_pca_75 = calculate_frobenius_norm(cov_matrix, pca_75)
    print(f"PCA 75% - Frobenius Norm: {frobenius_pca_75:.6f}, Time: {pca_75_time:.4f} seconds")
    
    # PCA 50% Simulation
    start_time = time.time()
    pca_50 = simulate_pca(cov_matrix, explained_variance_ratio=0.50)
    pca_50_time = time.time() - start_time
    frobenius_pca_50 = calculate_frobenius_norm(cov_matrix, pca_50)
    print(f"PCA 50% - Frobenius Norm: {frobenius_pca_50:.6f}, Time: {pca_50_time:.4f} seconds")
