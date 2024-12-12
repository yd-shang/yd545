import numpy as np

def expW(m, lam):
    """
    Calculate exponential weights.
    
    Parameters:
        m (int): Number of observations.
        lam (float): Decay factor (lambda).
        
    Returns:
        np.ndarray: Normalized exponential weights.
    """
    w = np.empty(m)
    for i in range(m):
        w[i] = (1 - lam) * lam**(m - i - 1)
    # Normalize weights to sum to 1
    return w / np.sum(w)

def ewCovar(x, lam):
    """
    Calculate exponentially weighted covariance matrix.
    
    Parameters:
        x (np.ndarray): Input matrix where rows are observations and columns are variables.
        lam (float): Decay factor (lambda).
        
    Returns:
        np.ndarray: Exponentially weighted covariance matrix.
    """
    m, n = x.shape
    w = expW(m, lam)  # Calculate weights
    
    # Remove the weighted mean from each series and scale by the square root of weights
    weighted_mean = np.dot(w, x)  # Weighted mean for each column
    xm = np.sqrt(w)[:, None] * (x - weighted_mean)  # Element-wise operation
    
    # Calculate the weighted covariance
    return np.dot(xm.T, xm)

def PCA_pctExplained(cov_matrix):
    """
    Calculate the percentage of variance explained by eigenvalues (PCA).
    
    Parameters:
        cov_matrix (np.ndarray): Covariance matrix.
        
    Returns:
        np.ndarray: Cumulative percentage of variance explained by the eigenvalues.
    """
    # Compute eigenvalues and sort in descending order
    eigenvalues = np.linalg.eigvalsh(cov_matrix)[::-1]  # Sorted in descending order
    
    # Calculate cumulative percentage of variance explained
    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues) / total_variance
    return cumulative_variance
