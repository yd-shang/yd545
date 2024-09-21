import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'DailyReturn.csv'
daily_return_data = pd.read_csv(file_path)

def exponentially_weighted_covariance(returns_data, lambda_factor):
    returns = returns_data.values
    num_assets = returns.shape[1]
    
    ew_cov_matrix = np.zeros((num_assets, num_assets))
    
    mean_returns = np.zeros(num_assets)
    
    for i in range(returns.shape[0]):
        deviation = returns[i] - mean_returns
        
        ew_cov_matrix = lambda_factor * ew_cov_matrix + (1 - lambda_factor) * np.outer(deviation, deviation)
        
        mean_returns = lambda_factor * mean_returns + (1 - lambda_factor) * returns[i]
    
    return ew_cov_matrix

def custom_pca(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    return sorted_eigenvalues, sorted_eigenvectors

def plot_cumulative_variance(eigenvalues, lambda_value):
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
    plt.title(f'Cumulative Variance Explained by PCA (λ={lambda_value})')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()

lambda_values = [0.9, 0.95, 0.97, 0.99]

for lambda_val in lambda_values:
    ew_cov_matrix = exponentially_weighted_covariance(daily_return_data, lambda_val)
    eigenvalues, eigenvectors = custom_pca(ew_cov_matrix)
    plot_cumulative_variance(eigenvalues, lambda_val)

