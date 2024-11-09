import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from statsmodels.tsa.ar_model import AutoReg
from datetime import datetime

daily_prices = pd.read_csv('DailyPrices.csv')
portfolio_data = pd.read_csv('problem2.csv')
problem3_data = portfolio_data

current_aapl_price = 165
risk_free_rate = 0.0425
dividend_rate = 1 / 165
current_date = datetime.strptime("2023-03-03", "%Y-%m-%d")

# Function to calculate implied volatility using Brent's method
def calculate_implied_volatility_brent(option_type, S, K, T, r, q, market_price):
    def bs_price(vol):
        d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        if option_type == "Call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - market_price
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1) - market_price

    try:
        implied_vol = brentq(bs_price, 1e-6, 5)
    except ValueError:
        implied_vol = np.nan
    return implied_vol

# Calculate implied volatilities for options in the portfolio
problem3_data['ImpliedVolatility'] = np.nan
for index, row in problem3_data.iterrows():
    if not pd.isna(row['OptionType']):
        expiration_date = datetime.strptime(row['ExpirationDate'], "%m/%d/%Y")
        T = (expiration_date - current_date).days / 365.0  # Time to maturity in years
        if T > 0:
            implied_volatility = calculate_implied_volatility_brent(
                option_type=row['OptionType'],
                S=current_aapl_price,
                K=row['Strike'],
                T=T,
                r=risk_free_rate,
                q=dividend_rate,
                market_price=row['CurrentPrice']
            )
            problem3_data.at[index, 'ImpliedVolatility'] = implied_volatility

# Function to calculate portfolio value
def calculate_portfolio_value(aapl_prices, data, r, q, T):
    portfolio_values = []
    for S in aapl_prices:
        portfolio_value = 0
        for _, row in data.iterrows():
            if pd.isna(row['OptionType']): 
                portfolio_value += row['Holding'] * (S - row['CurrentPrice'])
            else:
                K = row['Strike']
                vol = row['ImpliedVolatility']
                if T > 0 and vol > 0:
                    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
                    d2 = d1 - vol * np.sqrt(T)
                    if row['OptionType'] == "Call":
                        option_value = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                    else:
                        option_value = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
                    portfolio_value += row['Holding'] * (option_value - row['CurrentPrice'])
        portfolio_values.append(portfolio_value)
    return portfolio_values

# Setting expiration time and generating simulated AAPL prices
T_expiration = (datetime.strptime("04/21/2023", "%m/%d/%Y") - current_date).days / 365.0
aapl_log_returns = np.log(daily_prices['AAPL'] / daily_prices['AAPL'].shift(1)).dropna()
std_dev = aapl_log_returns.std()
num_simulations = 10000
simulated_returns = norm.rvs(loc=0, scale=std_dev, size=num_simulations)
simulated_aapl_prices = [current_aapl_price * np.exp(np.sum(simulated_returns[:i])) for i in range(1, 11)]

# Calculate portfolio values for each simulated AAPL price and compute risk metrics
portfolio_values_by_price = {}
for portfolio_name in problem3_data['Portfolio'].unique():
    portfolio_data = problem3_data[problem3_data['Portfolio'] == portfolio_name]
    portfolio_values = calculate_portfolio_value(simulated_aapl_prices, portfolio_data, risk_free_rate, dividend_rate, T_expiration)
    portfolio_values_by_price[portfolio_name] = portfolio_values

# Aggregate and calculate VaR, ES, Mean, Standard Deviation, Min, and Max for each portfolio
confidence_level = 0.95
portfolio_metrics = {}
for portfolio_name, values in portfolio_values_by_price.items():
    values = np.array(values)
    mean_value = values.mean()
    var_95 = np.percentile(values, (1 - confidence_level) * 100)
    es_95 = values[values <= var_95].mean()
    std_dev = values.std()
    min_value = values.min()
    max_value = values.max()
    
    portfolio_metrics[portfolio_name] = {
        'Mean': mean_value,
        'VaR95': abs(var_95),
        'ES95': abs(es_95),
        'Standard_Dev': std_dev,
        'Min': min_value,
        'Max': max_value
    }

# Display the results in a DataFrame
portfolio_metrics_df = pd.DataFrame(portfolio_metrics).T
print(portfolio_metrics_df)
