import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize

# Class to store regression results
class RegressionResult:
    def __init__(self, beta, sigma, df=None, errors=None):
        """
        :param beta: Coefficients
        :param sigma: Standard deviation (Normal or T)
        :param df: Degrees of freedom (for T distribution)
        :param errors: Residual errors
        """
        self.beta = beta
        self.sigma = sigma
        self.df = df
        self.errors = errors

# Ordinary Least Squares
def fit_ols(y, X):
    """
    Fit a regression model using Ordinary Least Squares (OLS).
    :param y: Response variable (vector)
    :param X: Predictor matrix (without intercept)
    :return: RegressionResult object with beta and errors
    """
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    errors = y - X @ beta
    return RegressionResult(beta=beta, sigma=np.std(errors), errors=errors)

# MLE for Normal distribution
def mle_normal(params, y, X):
    beta = params[:-1]
    sigma = params[-1]
    errors = y - X @ beta
    log_likelihood = -0.5 * len(errors) * np.log(2 * np.pi * sigma**2) - np.sum(errors**2) / (2 * sigma**2)
    return -log_likelihood

def fit_regression_mle(y, X):
    """
    Fit a regression model using Maximum Likelihood Estimation (MLE) for a Normal distribution.
    :param y: Response variable (vector)
    :param X: Predictor matrix (without intercept)
    :return: RegressionResult object with beta, sigma, and errors
    """
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term
    initial_params = np.append(np.zeros(X.shape[1]), 1)  # Initial guesses for beta and sigma
    result = minimize(mle_normal, initial_params, args=(y, X), method='L-BFGS-B', bounds=[(None, None)] * (X.shape[1]) + [(1e-6, None)])
    
    beta = result.x[:-1]
    sigma = result.x[-1]
    errors = y - X @ beta
    return RegressionResult(beta=beta, sigma=sigma, errors=errors)

# MLE for T distribution
def mle_t(params, y, X):
    beta = params[:-2]
    sigma = params[-2]
    df = params[-1]
    errors = y - X @ beta
    log_likelihood = np.sum(t.logpdf(errors / sigma, df=df) - np.log(sigma))
    return -log_likelihood

def fit_regression_t(y, X):
    """
    Fit a regression model using Maximum Likelihood Estimation (MLE) for a T distribution.
    :param y: Response variable (vector)
    :param X: Predictor matrix (without intercept)
    :return: RegressionResult object with beta, sigma, df, and errors
    """
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term
    initial_params = np.append(np.zeros(X.shape[1]), [1, 5])  # Initial guesses for beta, sigma, and df
    bounds = [(None, None)] * (X.shape[1]) + [(1e-6, None), (1e-6, None)]
    result = minimize(mle_t, initial_params, args=(y, X), method='L-BFGS-B', bounds=bounds)
    
    beta = result.x[:-2]
    sigma = result.x[-2]
    df = result.x[-1]
    errors = y - X @ beta
    return RegressionResult(beta=beta, sigma=sigma, df=df, errors=errors)

def fit_general_t(data):
    """
    Fit a T distribution to the given data using Maximum Likelihood Estimation (MLE).

    Parameters:
        data (array-like): Input data to fit (1-dimensional array).

    Returns:
        tuple:
            - mean (float): Mean of the fitted T distribution.
            - scale (float): Scale (standard deviation) of the fitted T distribution.
            - df (float): Degrees of freedom of the fitted T distribution.
            - t_distribution (scipy.stats.t object): Fitted T distribution object.
    """
    # Compute the sample mean and standard deviation
    mean = np.mean(data)
    scale = np.std(data, ddof=1)  # Sample standard deviation

    # Initial guess for degrees of freedom
    initial_df = 5.0

    # Define the negative log-likelihood function
    def neg_log_likelihood(params):
        df = params[0]
        if df <= 0:
            return np.inf
        return -np.sum(t.logpdf((data - mean) / scale, df=df))

    # Use `minimize` to find the best-fitting degrees of freedom
    result = minimize(
        neg_log_likelihood,
        x0=[initial_df],
        bounds=[(1e-6, None)],  # df must be positive
        method="L-BFGS-B"
    )

    # Extract the optimized degrees of freedom
    df = result.x[0]

    # Create a T distribution object using the fitted parameters
    t_distribution = t(df=df, loc=mean, scale=scale)

    return mean, scale, df, t_distribution

