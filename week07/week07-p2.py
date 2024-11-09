import pandas as pd
import numpy as np
from scipy.stats import norm

# Load data
daily_prices = pd.read_csv('DailyPrices.csv')
portfolio_data = pd.read_csv('problem2.csv')

# Step 1: Calculate daily log returns for AAPL
daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])
daily_prices['AAPL_Log_Return'] = np.log(daily_prices['AAPL'] / daily_prices['AAPL'].shift(1))
aapl_log_returns = daily_prices['AAPL_Log_Return'].dropna()

# Step 2: Fit a normal distribution to AAPL log returns and simulate log returns
std_dev = aapl_log_returns.std()
num_simulations = 1000  # Reduced to 1000 for faster execution
simulated_log_returns = norm.rvs(loc=0, scale=std_dev, size=num_simulations)

# Step 3: Apply these log returns to simulate AAPL prices
initial_price = 165  # Initial price as specified
simulated_prices = [initial_price]
for log_ret in simulated_log_returns:
    new_price = simulated_prices[-1] * np.exp(log_ret)
    simulated_prices.append(new_price)

# Remove the initial price from the list as we only want the simulated prices
simulated_prices = simulated_prices[1:]

# Parameters for Black-Scholes and portfolio calculation
risk_free_rate = 4.25 / 100
dividend_yield = 1 / 165  # Dividend yield based on $1.00 dividend and initial stock price
current_date = pd.Timestamp('2023-03-03')

# Function to calculate Black-Scholes option price with dividend yield
def black_scholes_price(S, K, T, r, sigma, q, option_type):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'Put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return 0

# Step 4: Calculate portfolio metrics for each unique portfolio in problem2.csv
portfolio_metrics_final = []

for portfolio_name in portfolio_data['Portfolio'].unique():
    portfolio_rows = portfolio_data[portfolio_data['Portfolio'] == portfolio_name]
    portfolio_values = []
    
    for S in simulated_prices:
        portfolio_value = 0
        for _, row in portfolio_rows.iterrows():
            holding = row['Holding']
            current_price = row['CurrentPrice']
            
            if pd.isna(row['OptionType']): 
                # Treat as stock, calculate P&L as holding * (simulated price - current price)
                portfolio_value += holding * (S - current_price)
            else:
                # Treat as option, use Black-Scholes with dividend yield
                K = row['Strike']
                implied_volatility = row['ImpliedVolatility'] if 'ImpliedVolatility' in row else std_dev
                expiration_date = pd.to_datetime(row['ExpirationDate'])
                time_to_expiry = (expiration_date - current_date).days / 365

                # Calculate option value using Black-Scholes
                option_value = black_scholes_price(S, K, time_to_expiry, risk_free_rate, implied_volatility, dividend_yield, row['OptionType'])
                
                # Accumulate portfolio value for option by subtracting current price
                portfolio_value += holding * (option_value - current_price)

        # Append calculated portfolio value for this price
        portfolio_values.append(portfolio_value)
    
    # Calculate aggregated risk metrics for the portfolio
    pnl_array = np.array(portfolio_values)
    # Calculate the currentValue correctly by summing individual holding * current price
    current_value = (portfolio_rows['Holding'] * portfolio_rows['CurrentPrice']).sum()
    
    # Calculating VaR and ES as positive values (absolute value of losses)
    VaR95 = np.percentile(pnl_array, 5)
    ES95 = pnl_array[pnl_array <= VaR95].mean()
    VaR99 = np.percentile(pnl_array, 1)
    ES99 = pnl_array[pnl_array <= VaR99].mean()
    
    agg_metrics = {
        'Portfolio': portfolio_name,
        'currentValue': current_value,
        'VaR95': abs(VaR95),  
        'ES95': abs(ES95), 
        'VaR99': abs(VaR99), 
        'ES99': abs(ES99),  
        'Standard_Dev': pnl_array.std(),
        'min': pnl_array.min(),
        'max': pnl_array.max(),
        'mean': pnl_array.mean()
    }
    portfolio_metrics_final.append(agg_metrics)


portfolio_metrics_final_df = pd.DataFrame(portfolio_metrics_final)
portfolio_metrics_final_df.to_csv("corrected_portfolio_risk_metrics.csv", index=False)

# Displaying the results
print(portfolio_metrics_final_df)



