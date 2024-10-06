import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.stats as stats 

portfolio_file_path = 'portfolio.csv'
prices_file_path = 'DailyPrices.csv'

portfolio_df = pd.read_csv(portfolio_file_path)
prices_df = pd.read_csv(prices_file_path)

returns_df = prices_df.drop(columns=['Date']).pct_change().dropna()

#EWMA
def ewma_covariance(returns, lambda_=0.97):
    n, m = returns.shape
    cov_matrix = np.zeros((m, m))
    weights = np.array([(1 - lambda_) * lambda_ ** i for i in range(n)])[::-1]
    weights /= weights.sum()
    
    mean_returns = np.average(returns, axis=0, weights=weights)
    deviations = returns - mean_returns
    for i in range(n):
        cov_matrix += weights[i] * np.outer(deviations[i], deviations[i])
    
    return cov_matrix

#Prepare data
portfolio_stocks = portfolio_df['Stock'].unique()

available_stocks = [stock for stock in portfolio_stocks if stock in returns_df.columns]

portfolio_returns = returns_df[available_stocks]

#Calculate portfolio value
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

# Delta-Normal VaR & VaR
def delta_normal_var(holdings, covariance_matrix, portfolio_value, alpha=0.05):
    
    portfolio_weights = holdings / holdings.sum() 
    portfolio_variance = portfolio_weights.T @ covariance_matrix @ portfolio_weights
    portfolio_std = np.sqrt(portfolio_variance)
    var_percentage = stats.norm.ppf(alpha) * portfolio_std 
    var_dollar = portfolio_value * abs(var_percentage) 
    return var_dollar

def historical_var(returns, holdings, portfolio_value, alpha=0.05):
    portfolio_weights = holdings / holdings.sum()
    portfolio_returns = returns @ portfolio_weights
    var_percentage = np.percentile(portfolio_returns, 100 * alpha)
    var_dollar = portfolio_value * abs(var_percentage)
    return var_dollar

# Align
def align_holdings_and_returns(portfolio_df, returns_df):
    total_stocks = portfolio_df['Stock'].unique()
    aligned_stocks = [stock for stock in total_stocks if stock in returns_df.columns]
    aligned_holdings = np.concatenate([portfolio_df[(portfolio_df['Portfolio'] == portfolio) & (portfolio_df['Stock'].isin(aligned_stocks))]['Holding'].values 
                                   for portfolio in portfolio_df['Portfolio'].unique()])
    return aligned_holdings, aligned_stocks

portfolio_var_results = {}

for portfolio in portfolio_df['Portfolio'].unique():
    portfolio_stocks = portfolio_df[portfolio_df['Portfolio'] == portfolio]['Stock'].tolist()
    portfolio_holdings = portfolio_df[portfolio_df['Portfolio'] == portfolio]['Holding'].values
    portfolio_stocks = [stock for stock in portfolio_stocks if stock in portfolio_returns.columns]
    
    aligned_holdings = np.array([portfolio_df[(portfolio_df['Portfolio'] == portfolio) & (portfolio_df['Stock'] == stock)]['Holding'].values[0] for stock in portfolio_stocks])
    
    if len(portfolio_stocks) > 0:
        returns_matrix = portfolio_returns[portfolio_stocks].to_numpy()

        cov_matrix = ewma_covariance(returns_matrix)

        portfolio_value = portfolio_values[portfolio]

        var_delta_normal = delta_normal_var(aligned_holdings, cov_matrix, portfolio_value)

        var_historical = historical_var(returns_matrix, aligned_holdings, portfolio_value)

        portfolio_var_results[portfolio] = {
            'Delta Normal VaR ($)': var_delta_normal,
            'Historical VaR ($)': var_historical
        }

for portfolio, results in portfolio_var_results.items():
    print(f"Portfolio {portfolio} Delta Normal VaR in $: {results['Delta Normal VaR ($)']:.2f}")
    print(f"Portfolio {portfolio} Historical VaR in $: {results['Historical VaR ($)']:.2f}")

total_holdings, aligned_stocks = align_holdings_and_returns(portfolio_df, portfolio_returns)

aligned_returns_matrix = portfolio_returns[aligned_stocks].to_numpy()

total_cov_matrix = ewma_covariance(aligned_returns_matrix)

total_value = sum(portfolio_values.values())
total_var_delta_normal = delta_normal_var(total_holdings, total_cov_matrix, total_value)
total_var_historical = historical_var(aligned_returns_matrix, total_holdings, total_value)

print(f"Total Portfolio Delta Normal VaR in $: {total_var_delta_normal:.2f}")
print(f"Total Portfolio Historical VaR in $: {total_var_historical:.2f}")

