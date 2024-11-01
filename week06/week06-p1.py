import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

S = 165  
X = 165  
r = 5.25 / 100 
coupon_rate = 0.53 / 100
current_date = np.datetime64('2023-03-03')
expiration_date = np.datetime64('2023-03-17')

# Step 1: Calculate Time to Maturity
days_to_maturity = (expiration_date - current_date).astype(int)
T = days_to_maturity / 365  # Convert days to years

# Step 2: Function to calculate Call and Put option prices using Black-Scholes
def black_scholes_option_price(S, X, T, r, sigma, option_type='call'):
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Step 3: Calculate and store option prices for a range of implied volatilities
implied_volatilities = np.linspace(0.10, 0.80, 50)  # From 10% to 80%
call_prices = [black_scholes_option_price(S, X, T, r, sigma, 'call') for sigma in implied_volatilities]
put_prices = [black_scholes_option_price(S, X, T, r, sigma, 'put') for sigma in implied_volatilities]

# Step 4: Plot the results
plt.figure(figsize=(12, 6))
plt.plot(implied_volatilities, call_prices, label="Call Option Price", linestyle='-', marker='o')
plt.plot(implied_volatilities, put_prices, label="Put Option Price", linestyle='--', marker='x')
plt.xlabel("Implied Volatility")
plt.ylabel("Option Price")
plt.title("Option Prices vs. Implied Volatility (Call and Put)")
plt.legend()
plt.grid(True)
plt.show()
