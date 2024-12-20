import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from statsmodels.tsa.ar_model import AutoReg
from datetime import datetime

file_path = 'problem3.csv'
problem3_data = pd.read_csv(file_path)

current_aapl_price = 170.15
risk_free_rate = 0.0525
dividend_rate = 0.0057
current_date = datetime.strptime("2023-10-30", "%Y-%m-%d")

#1
# Function to calculate implied volatility
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

# Calculate implied volatilities for options
problem3_data['ImpliedVolatility'] = np.nan
for index, row in problem3_data.iterrows():
    if not pd.isna(row['OptionType']):
        expiration_date = datetime.strptime(row['ExpirationDate'], "%m/%d/%Y")
        T = (expiration_date - current_date).days / 365.0 
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

# portfolio value calculation function
def calculate_portfolio_value(portfolio_name, aapl_prices, data, r, q, T):
    portfolio_values = []
    
    for S in aapl_prices:
        portfolio_value = 0
        for _, row in data.iterrows():
            if pd.isna(row['OptionType']): 
                portfolio_value += row['Holding'] * (S - row['CurrentPrice'])
            else:
                K = row['Strike']
                d1 = (np.log(S / K) + (r - q + 0.5 * row['ImpliedVolatility'] ** 2) * T) / (row['ImpliedVolatility'] * np.sqrt(T))
                d2 = d1 - row['ImpliedVolatility'] * np.sqrt(T)
                if row['OptionType'] == "Call":
                    option_value = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                else:
                    option_value = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
                portfolio_value += row['Holding'] * option_value 

        portfolio_values.append(portfolio_value)
    
    return portfolio_values

# Define the range of AAPL prices for the simulation
aapl_price_range = np.linspace(100, 250, 100) 
T_expiration = (datetime.strptime("12/15/2023", "%m/%d/%Y") - current_date).days / 365.0

# Plot portfolio values
plt.figure(figsize=(12, 8))
unique_portfolios = problem3_data['Portfolio'].unique()

for portfolio_name in unique_portfolios:
    portfolio_data = problem3_data[problem3_data['Portfolio'] == portfolio_name]
    portfolio_values = calculate_portfolio_value(portfolio_name, aapl_price_range, portfolio_data, risk_free_rate, dividend_rate, T_expiration)
    plt.plot(aapl_price_range, portfolio_values, label=portfolio_name)

plt.title("Portfolio Values Over A Range of AAPL Underlying Prices")
plt.xlabel("AAPL Underlying Price ($)")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.show()

#2 - portfolio
current_date = datetime.strptime("2023-10-30", "%Y-%m-%d")
daily_prices_df = pd.read_csv('DailyPrices.csv')
problem3_df = pd.read_csv('problem3.csv')
num_simulation_days = 10
confidence_level = 0.95

# 1. Calculate log returns for AAPL and demean the series
aapl_prices = pd.to_numeric(daily_prices_df['AAPL'], errors='coerce').dropna()
aapl_log_returns = np.log(aapl_prices / aapl_prices.shift(1)).dropna()
aapl_log_returns_demeaned = aapl_log_returns - aapl_log_returns.mean()

# 2. Fit AR(1) model to the demeaned log returns
ar_model = AutoReg(aapl_log_returns_demeaned, lags=1).fit()
alpha, beta = ar_model.params[0], ar_model.params[1]
sigma = ar_model.resid.std()

# 3. Simulate 10 days of returns and calculate AAPL prices
simulated_returns = [0]
for _ in range(num_simulation_days):
    new_return = alpha + beta * simulated_returns[-1] + np.random.normal(0, sigma)
    simulated_returns.append(new_return)

simulated_aapl_prices = [current_aapl_price * np.exp(np.sum(simulated_returns[:i])) for i in range(1, num_simulation_days + 1)]

# 4. Define implied volatility and Black-Scholes pricing functions
def calculate_implied_volatility_brent(option_type, S, K, T, r, q, market_price):
    def bs_price(vol):
        d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        if option_type == "Call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - market_price
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1) - market_price

    try:
        return brentq(bs_price, 1e-6, 5)
    except ValueError:
        return np.nan

def calculate_option_price(option_type, S, K, T, r, q, vol):
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if option_type == "Call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

# 5. Calculate portfolio values for each simulated AAPL price
portfolio_values_by_price = {}
for portfolio_name in problem3_df['Portfolio'].unique():
    portfolio_data = problem3_df[problem3_df['Portfolio'] == portfolio_name]
    portfolio_values = []
    
    for S in simulated_aapl_prices:
        portfolio_value = 0
        for _, row in portfolio_data.iterrows():
            if pd.isna(row['OptionType']):  # For stocks
                portfolio_value += row['Holding'] * (S - row['CurrentPrice'])
            else:  # For options
                K = row['Strike']
                T = (datetime.strptime(row['ExpirationDate'], "%m/%d/%Y") - current_date).days / 365.0
                implied_vol = calculate_implied_volatility_brent(
                    option_type=row['OptionType'],
                    S=current_aapl_price,
                    K=K,
                    T=T,
                    r=risk_free_rate,
                    q=dividend_rate,
                    market_price=row['CurrentPrice']
                )
                if T > 0 and not pd.isna(implied_vol):
                    option_value = calculate_option_price(
                        option_type=row['OptionType'],
                        S=S,
                        K=K,
                        T=T,
                        r=risk_free_rate,
                        q=dividend_rate,
                        vol=implied_vol
                    )
                    portfolio_value += row['Holding'] * option_value

        portfolio_values.append(portfolio_value)
    
    portfolio_values_by_price[portfolio_name] = portfolio_values

# 6. Calculate Mean, VaR, and ES for each portfolio
portfolio_metrics = {}
for portfolio_name, values in portfolio_values_by_price.items():
    mean_value = np.mean(values)
    var_95 = np.percentile(values, (1 - confidence_level) * 100)
    es_95 = np.mean([v for v in values if v <= var_95])
    
    portfolio_metrics[portfolio_name] = {'Mean': mean_value, 'VaR': var_95, 'ES': es_95}

# Display the results
portfolio_metrics_df = pd.DataFrame(portfolio_metrics).T
print(portfolio_metrics_df)




#2 - only aapl
data = pd.read_csv('/Users/shangyudi/Desktop/Duke/2024fall/quantitative risk management/week06/DailyPrices.csv')
# Calculate log returns of AAPL
data['AAPL_Log_Return'] = np.log(data['AAPL'] / data['AAPL'].shift(1))
aapl_log_returns = data['AAPL_Log_Return'].dropna()

# Demean the log returns
demeaned_returns = aapl_log_returns - aapl_log_returns.mean()

# Fit an AR(1) model to the demeaned returns
ar_model = AutoReg(demeaned_returns, lags=1).fit()

# Simulate 10 days ahead using the AR(1) model
num_simulations = 1000  # Number of paths for simulation
current_price = 170.15
simulated_returns = []

for _ in range(num_simulations):
    simulated_path = [current_price]
    last_return = demeaned_returns.iloc[-1]
    
    for _ in range(10):
        next_return = ar_model.params['const'] + ar_model.params['AAPL_Log_Return.L1'] * last_return + np.random.normal(scale=ar_model.sigma2**0.5)
        simulated_price = simulated_path[-1] * np.exp(next_return)
        simulated_path.append(simulated_price)
        last_return = next_return
    
    simulated_returns.append(simulated_path[-1])

# Calculate Mean, VaR (5% level), and ES (Expected Shortfall at 5% level)
mean_price = np.mean(simulated_returns)
var_95 = np.percentile(simulated_returns, 5)
es_95 = np.mean([x for x in simulated_returns if x <= var_95])

print(mean_price, var_95, es_95)
