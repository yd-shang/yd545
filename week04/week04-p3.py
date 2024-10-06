import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.stats as stats 

portfolio_file_path = 'portfolio.csv'
prices_file_path = 'DailyPrices.csv'

portfolio_df = pd.read_csv(portfolio_file_path)
prices_df = pd.read_csv(prices_file_path)

# Calculate returns using both arithmetic and log returns
def return_calculate(prices: pd.DataFrame, method="DISCRETE", date_column="Date"):
    vars = prices.columns.tolist()
    if date_column in vars:
        vars.remove(date_column)
    p = prices[vars].to_numpy()
    n, m = p.shape
    p2 = np.zeros((n-1, m))

    if method.upper() == "DISCRETE":
        p2 = (p[1:, :] / p[:-1, :]) - 1.0  # Discrete (arithmetic) returns
    elif method.upper() == "LOG":
        p2 = np.log(p[1:, :] / p[:-1, :])  # Log returns
    else:
        raise ValueError("method must be 'DISCRETE' or 'LOG'")
    
    dates = prices[date_column].iloc[1:].reset_index(drop=True)
    returns_df = pd.DataFrame(p2, columns=vars)
    returns_df.insert(0, date_column, dates)
    
    return returns_df

discrete_returns_df = return_calculate(prices_df, method="DISCRETE")
log_returns_df = return_calculate(prices_df, method="LOG")

# Exponentially Weighted Covariance Matrix (EWMA) function
def ewma_covariance(returns, lambda_=0.97):
    n, m = returns.shape
    cov_matrix = np.zeros((m, m))
    weights = np.array([(1 - lambda_) * lambda_ ** i for i in range(n)])[::-1]
    weights /= weights.sum()  # Normalize weights
    
    mean_returns = np.average(returns, axis=0, weights=weights)
    deviations = returns - mean_returns
    for i in range(n):
        cov_matrix += weights[i] * np.outer(deviations[i], deviations[i])
    
    return cov_matrix

# Fetch all stocks in the portfolio
portfolio_stocks = portfolio_df['Stock'].unique()

# Filter out the stocks present in DailyPrices
available_stocks = [stock for stock in portfolio_stocks if stock in discrete_returns_df.columns]

# Filter the returns for available stocks for both discrete and log returns
discrete_portfolio_returns = discrete_returns_df[available_stocks]
log_portfolio_returns = log_returns_df[available_stocks]

# Calculate portfolio value
def calculate_portfolio_value(portfolio_df, latest_prices):
    portfolio_values = {}
    unique_portfolios = portfolio_df['Portfolio'].unique()

    for portfolio in unique_portfolios:
        portfolio_holdings = portfolio_df[portfolio_df['Portfolio'] == portfolio]
        total_value = 0
        for _, row in portfolio_holdings.iterrows():
            stock = row['Stock']
            shares = row['Holding']
            if stock in latest_prices.index:
                stock_price = latest_prices[stock]
                total_value += stock_price * shares
        portfolio_values[portfolio] = total_value
    return portfolio_values


latest_prices = prices_df.iloc[-1]
portfolio_values = calculate_portfolio_value(portfolio_df, latest_prices)

# EW VaR
def ew_var(holdings, ew_cov_matrix, portfolio_value, alpha=0.05):
    portfolio_weights = holdings / holdings.sum() 
    portfolio_variance = portfolio_weights.T @ ew_cov_matrix @ portfolio_weights
    portfolio_std = np.sqrt(portfolio_variance)
    var_percentage = stats.norm.ppf(alpha) * portfolio_std 
    var_dollar = portfolio_value * abs(var_percentage) 
    return var_dollar

# Align holdings and returns
def align_holdings_and_returns(portfolio_df, returns_df):
    total_stocks = portfolio_df['Stock'].unique()
    aligned_stocks = [stock for stock in total_stocks if stock in returns_df.columns]
    
    aligned_holdings = np.concatenate([portfolio_df[(portfolio_df['Portfolio'] == portfolio) & (portfolio_df['Stock'].isin(aligned_stocks))]['Holding'].values 
                                   for portfolio in portfolio_df['Portfolio'].unique()])
    
    return aligned_holdings, aligned_stocks

# Calculate and compare EW VaR
portfolio_var_results = {}

for portfolio in portfolio_df['Portfolio'].unique():
    portfolio_stocks = portfolio_df[portfolio_df['Portfolio'] == portfolio]['Stock'].tolist()
    portfolio_holdings = portfolio_df[portfolio_df['Portfolio'] == portfolio]['Holding'].values

    portfolio_stocks = [stock for stock in portfolio_stocks if stock in discrete_portfolio_returns.columns]
    
    aligned_holdings = np.array([portfolio_df[(portfolio_df['Portfolio'] == portfolio) & (portfolio_df['Stock'] == stock)]['Holding'].values[0] for stock in portfolio_stocks])
    
    if len(portfolio_stocks) > 0:
        # Discrete returns
        discrete_returns_matrix = discrete_portfolio_returns[portfolio_stocks].to_numpy()
        ew_cov_matrix_discrete = ewma_covariance(discrete_returns_matrix)
        portfolio_value = portfolio_values[portfolio]
        var_ew_discrete = ew_var(aligned_holdings, ew_cov_matrix_discrete, portfolio_value)

        # Log returns
        log_returns_matrix = log_portfolio_returns[portfolio_stocks].to_numpy()
        ew_cov_matrix_log = ewma_covariance(log_returns_matrix)
        var_ew_log = ew_var(aligned_holdings, ew_cov_matrix_log, portfolio_value)

        portfolio_var_results[portfolio] = {
            'EW VaR (Discrete) in $': var_ew_discrete,
            'EW VaR (Log) in $': var_ew_log
        }

for portfolio, results in portfolio_var_results.items():
    print(f"Portfolio {portfolio} EW VaR (Discrete) in $: {results['EW VaR (Discrete) in $']:.2f}")
    print(f"Portfolio {portfolio} EW VaR (Log) in $: {results['EW VaR (Log) in $']:.2f}")

total_holdings, aligned_stocks = align_holdings_and_returns(portfolio_df, discrete_portfolio_returns)
aligned_discrete_returns_matrix = discrete_portfolio_returns[aligned_stocks].to_numpy()
aligned_log_returns_matrix = log_portfolio_returns[aligned_stocks].to_numpy()

total_cov_matrix_discrete = ewma_covariance(aligned_discrete_returns_matrix)
total_cov_matrix_log = ewma_covariance(aligned_log_returns_matrix)

total_value = sum(portfolio_values.values())

total_var_ew_discrete = ew_var(total_holdings, total_cov_matrix_discrete, total_value)
total_var_ew_log = ew_var(total_holdings, total_cov_matrix_log, total_value)

print(f"Total Portfolio EW VaR (Discrete) in $: {total_var_ew_discrete:.2f}")
print(f"Total Portfolio EW VaR (Log) in $: {total_var_ew_log:.2f}")
