import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import brentq

file_path = 'AAPL_Options.csv'
options_data = pd.read_csv(file_path)

current_price = 170.15 
risk_free_rate = 5.25 / 100 
dividend_rate = 0.57 / 100 
current_date = np.datetime64('2023-10-30') 

options_data['Expiration'] = pd.to_datetime(options_data['Expiration'])
options_data['Time_to_Maturity'] = (options_data['Expiration'] - current_date).dt.days / 365

# Black-Scholes formula function to calculate the option price
def black_scholes_price(S, X, T, r, q, sigma, option_type='call'):
    d1 = (np.log(S / X) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return X * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

# Function to calculate implied volatility
def implied_volatility(S, X, T, r, q, market_price, option_type='call'):
    try:
        implied_vol = brentq(lambda sigma: black_scholes_price(S, X, T, r, q, sigma, option_type) - market_price, 0.01, 5.0)
    except ValueError:
        implied_vol = np.nan
    return implied_vol

# Calculate implied volatility for each option
options_data['Implied_Volatility'] = options_data.apply(
    lambda row: implied_volatility(
        S=current_price,
        X=row['Strike'],
        T=row['Time_to_Maturity'],
        r=risk_free_rate,
        q=dividend_rate,
        market_price=row['Last Price'],
        option_type=row['Type'].lower()
    ), axis=1
)

# Separate the data into call and put options
call_options = options_data[options_data['Type'] == 'Call']
put_options = options_data[options_data['Type'] == 'Put']

options_data[['Type', 'Strike', 'Last Price', 'Time_to_Maturity', 'Implied_Volatility']].head()

# Plotting Implied Volatility vs. Strike Price for Calls and Puts
plt.figure(figsize=(12, 6))

plt.plot(call_options['Strike'], call_options['Implied_Volatility'], label='Call Options', linestyle='-', marker='o')
plt.plot(put_options['Strike'], put_options['Implied_Volatility'], label='Put Options', linestyle='--', marker='x')

plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatility vs. Strike Price for AAPL Options")
plt.legend()
plt.grid(True)
plt.show()
