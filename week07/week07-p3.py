import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

daily_prices = pd.read_csv('DailyPrices.csv', index_col='Date', parse_dates=True)
momentum_factor = pd.read_csv('/F-F_Momentum_Factor_daily.CSV', index_col='Date', parse_dates=True)
ff_factors = pd.read_csv('F-F_Research_Data_Factors_daily.CSV', index_col='Date', parse_dates=True)


momentum_factor.columns = ['MOM']
ff_factors.columns = ['MKT_RF', 'SMB', 'HML', 'RF']
momentum_factor.index = pd.to_datetime(momentum_factor.index, format='%Y%m%d', errors='coerce')
ff_factors.index = pd.to_datetime(ff_factors.index, format='%Y%m%d', errors='coerce')
momentum_factor.dropna(inplace=True)
ff_factors.dropna(inplace=True)

# Filter dates
common_dates = daily_prices.index.intersection(ff_factors.index).intersection(momentum_factor.index)
daily_prices = daily_prices.loc[common_dates]
ff_factors = ff_factors.loc[common_dates]
momentum_factor = momentum_factor.loc[common_dates]

# Calculate log returns and excess returns
log_returns = np.log(daily_prices / daily_prices.shift(1)).dropna()
factors = ff_factors.join(momentum_factor).dropna()
factors[['MKT_RF', 'SMB', 'HML', 'RF', 'MOM']] /= 100
excess_returns = log_returns.sub(factors['RF'], axis=0)

# Define stocks for the analysis
selected_stocks = [
    'AAPL', 'META', 'UNH', 'MA', 'MSFT', 'NVDA', 'HD', 'PFE', 'AMZN', 'BRK-B',
    'PG', 'XOM', 'TSLA', 'JPM', 'V', 'DIS', 'GOOGL', 'JNJ', 'BAC', 'CSCO'
]
clean_excess_returns = excess_returns[selected_stocks].dropna()
aligned_factors = factors.loc[clean_excess_returns.index]

# Fit the 4-factor model and calculate expected returns
def fit_four_factor_model(stock_returns, factors):
    X = sm.add_constant(factors[['MKT_RF', 'SMB', 'HML', 'MOM']])
    model = sm.OLS(stock_returns, X).fit()
    return model

expected_daily_returns = {}
for stock in selected_stocks:
    model = fit_four_factor_model(clean_excess_returns[stock], aligned_factors)
    expected_daily_returns[stock] = (
        model.params['const'] + model.params['MKT_RF'] * aligned_factors['MKT_RF'].mean() +
        model.params['SMB'] * aligned_factors['SMB'].mean() +
        model.params['HML'] * aligned_factors['HML'].mean() +
        model.params['MOM'] * aligned_factors['MOM'].mean()
    )

# Annualize expected returns
annual_expected_returns = {stock: ((1 + ret) ** 252 - 1) for stock, ret in expected_daily_returns.items()}

# Print the expected annual returns for the 20 stocks
print("Expected Annual Returns for Selected Stocks:")
for stock, rate in annual_expected_returns.items():
    print(f"{stock}: {rate:.4f}")
    
# Calculate the annual covariance matrix
daily_cov_matrix = log_returns[selected_stocks].cov()
annual_cov_matrix = daily_cov_matrix * 252

# Print the covariance matrix
print("\nAnnual Covariance Matrix:")
print(annual_cov_matrix)

# Optimize the portfolio for maximum Sharpe Ratio
risk_free_rate = 0.05
mu = np.array(list(annual_expected_returns.values()))
cov_matrix = annual_cov_matrix.to_numpy()
num_assets = len(selected_stocks)

def negative_sharpe_ratio(weights):
    portfolio_return = np.dot(weights, mu)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(portfolio_return - risk_free_rate) / portfolio_volatility

constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
initial_weights = np.array([1.0 / num_assets] * num_assets)

result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x
optimal_weights_percentage = {stock: weight * 100 for stock, weight in zip(selected_stocks, optimal_weights)}
optimal_return = np.dot(optimal_weights, mu)
optimal_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
optimal_sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility

# Display results
{
    "Optimal Weights (%)": optimal_weights_percentage,
    "Portfolio Expected Return": optimal_return,
    "Portfolio Volatility": optimal_volatility,
    "Portfolio Sharpe Ratio": optimal_sharpe_ratio
}

# Print the optimal weights and portfolio metrics
print("\nOptimal Portfolio Weights (%):")
for stock, weight in optimal_weights_percentage.items():
    print(f"{stock}: {weight:.2f}%")

print("\nOptimal Portfolio Metrics:")
print(f"Portfolio Expected Return: {optimal_return:.4f}")
print(f"Portfolio Volatility: {optimal_volatility:.4f}")
print(f"Portfolio Sharpe Ratio: {optimal_sharpe_ratio:.4f}")
