import numpy as np
from scipy.stats import t, norm

def calculate_var_es(returns, confidence_level=0.05):
    # Sort the returns
    sorted_returns = np.sort(returns)
    # Value at Risk (VaR): 5th percentile
    var = -np.percentile(sorted_returns, confidence_level * 100)
    # Expected Shortfall (ES): Average of returns below the VaR threshold
    es = -sorted_returns[sorted_returns <= var].mean()
    return var, es