import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

risk_free_rate = 0.0475

# Calculate the mean returns and covariance matrix of the assets
mean_returns = data.mean().values
cov_matrix = data.cov().values

# Define the Sharpe Ratio function to maximize
def sharpe_ratio(weights):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(portfolio_return - risk_free_rate) / portfolio_volatility 

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = [(-1, 1) for _ in range(len(mean_returns))] 

# Initial guess for weights
initial_weights = np.array([1/3, 1/3, 1/3])

# Optimization for Sharpe Ratio
result_sharpe = minimize(sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights_sharpe = result_sharpe.x

# Calculate the portfolio metrics for Sharpe Ratio
portfolio_return_sharpe = np.dot(optimal_weights_sharpe, mean_returns)
portfolio_volatility_sharpe = np.sqrt(np.dot(optimal_weights_sharpe.T, np.dot(cov_matrix, optimal_weights_sharpe)))
sharpe_ratio_sharpe = (portfolio_return_sharpe - risk_free_rate) / portfolio_volatility_sharpe

# Function to calculate Expected Shortfall
def expected_shortfall(returns, weights, alpha=0.025):
    portfolio_returns = np.dot(returns, weights)
    sorted_returns = np.sort(portfolio_returns)
    cutoff_index = int(np.ceil(alpha * len(sorted_returns)))
    return -np.mean(sorted_returns[:cutoff_index])

# Define the new risk-adjusted return metric
def risk_adjusted_return(weights, alpha=0.025):
    portfolio_return = np.dot(weights, mean_returns)
    es_value = expected_shortfall(data.values, weights, alpha)
    return -(portfolio_return - risk_free_rate) / es_value 

# Optimization for the new risk-adjusted return metric
result_risk_adjusted = minimize(risk_adjusted_return, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights_risk_adjusted = result_risk_adjusted.x

# Calculate the portfolio metrics for the new metric
portfolio_return_risk_adjusted = np.dot(optimal_weights_risk_adjusted, mean_returns)
portfolio_volatility_risk_adjusted = np.sqrt(np.dot(optimal_weights_risk_adjusted.T, np.dot(cov_matrix, optimal_weights_risk_adjusted)))
es_risk_adjusted = expected_shortfall(data.values, optimal_weights_risk_adjusted)
risk_adjusted_ratio = (portfolio_return_risk_adjusted - risk_free_rate) / es_risk_adjusted

# Visualization: Portfolio Weights Comparison
labels = ['A1', 'A2', 'A3']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

# Plot the weights for both portfolios
bar1 = ax.bar(x - width/2, optimal_weights_sharpe, width, label='Sharpe Ratio Portfolio')
bar2 = ax.bar(x + width/2, optimal_weights_risk_adjusted, width, label='Risk-Adjusted Portfolio')

ax.set_xlabel('Assets')
ax.set_ylabel('Portfolio Weights')
ax.set_title('Comparison of Portfolio Weights')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()

portfolio_comparison = pd.DataFrame({
    "Metric": ["Expected Return", "Volatility", "Sharpe Ratio", "Expected Shortfall", "Risk-Adjusted Ratio"],
    "Sharpe Ratio Portfolio": [portfolio_return_sharpe, portfolio_volatility_sharpe, sharpe_ratio_sharpe, "-", "-"],
    "Risk-Adjusted Portfolio": [portfolio_return_risk_adjusted, portfolio_volatility_risk_adjusted, "-", es_risk_adjusted, risk_adjusted_ratio]
})

print(portfolio_comparison)

weights_comparison = pd.DataFrame({
    "Asset": ['A1', 'A2', 'A3'],
    "Sharpe Ratio Portfolio Weights": optimal_weights_sharpe,
    "Risk-Adjusted Portfolio Weights": optimal_weights_risk_adjusted
})


print("Asset Weights Comparison:")
print(weights_comparison)

