import numpy as np
import pandas as pd
from scipy.stats import norm


S = 151.03  
K = 165  
r = 0.0425 
q = 0.0053  
sigma = 0.20  
dividend = 0.88  
expiration_date = np.datetime64('2022-04-15')
current_date = np.datetime64('2022-03-13')
dividend_date = (np.datetime64('2022-04-11') - current_date).astype('timedelta64[D]').astype(int) / 365.0
n = 50  # Number of steps in the binomial tree

# Time to expiration in years
T = (expiration_date - current_date).astype('timedelta64[D]').astype(int) / 365.0

# Define d1 and d2 for Black-Scholes formula
def d1(S, K, T, r, q, sigma):
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, q, sigma):
    return d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)

# Black-Scholes call and put price functions
def call_price(S, K, T, r, q, sigma):
    D1 = d1(S, K, T, r, q, sigma)
    D2 = d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)

def put_price(S, K, T, r, q, sigma):
    D1 = d1(S, K, T, r, q, sigma)
    D2 = d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * np.exp(-q * T) * norm.cdf(-D1)

# Calculate Greeks
def call_delta(S, K, T, r, q, sigma):
    return np.exp(-q * T) * norm.cdf(d1(S, K, T, r, q, sigma))

def put_delta(S, K, T, r, q, sigma):
    return np.exp(-q * T) * (norm.cdf(d1(S, K, T, r, q, sigma)) - 1)

def gamma(S, K, T, r, q, sigma):
    D1 = d1(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(D1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, q, sigma):
    D1 = d1(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(D1) * np.sqrt(T)

def call_theta(S, K, T, r, q, sigma):
    D1 = d1(S, K, T, r, q, sigma)
    D2 = d2(S, K, T, r, q, sigma)
    term1 = -S * np.exp(-q * T) * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
    term2 = q * S * np.exp(-q * T) * norm.cdf(D1)
    term3 = r * K * np.exp(-r * T) * norm.cdf(D2)
    return term1 - term2 - term3

def put_theta(S, K, T, r, q, sigma):
    D1 = d1(S, K, T, r, q, sigma)
    D2 = d2(S, K, T, r, q, sigma)
    term1 = -S * np.exp(-q * T) * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
    term2 = q * S * np.exp(-q * T) * norm.cdf(-D1)
    term3 = r * K * np.exp(-r * T) * norm.cdf(-D2)
    return term1 + term2 - term3

def call_rho(S, K, T, r, q, sigma):
    D2 = d2(S, K, T, r, q, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(D2)

def put_rho(S, K, T, r, q, sigma):
    D2 = d2(S, K, T, r, q, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-D2)

# Binomial Tree Model for American Options with Dividend Adjustments
def binomial_tree_american_option(S, K, T, r, q, sigma, n, option_type='call', dividend_date=None, dividend_amount=0):
    dt = T / n  # Time step
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize stock price tree
    stock_price_tree = np.zeros((n + 1, n + 1))
    stock_price_tree[0, 0] = S

    for i in range(1, n + 1):
        for j in range(i + 1):
            stock_price_tree[j, i] = S * (u ** (i - j)) * (d ** j)
            if dividend_date and (i * dt >= dividend_date):
                stock_price_tree[j, i] -= dividend_amount

    # Initialize option value tree
    option_tree = np.zeros((n + 1, n + 1))
    if option_type == 'call':
        option_tree[:, n] = np.maximum(0, stock_price_tree[:, n] - K)
    elif option_type == 'put':
        option_tree[:, n] = np.maximum(0, K - stock_price_tree[:, n])

    # Backward induction for American option pricing
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            hold_value = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            exercise_value = max(0, (stock_price_tree[j, i] - K) if option_type == 'call' else (K - stock_price_tree[j, i]))
            option_tree[j, i] = max(hold_value, exercise_value)

    return option_tree[0, 0]

# Calculate Greeks for Black-Scholes Model
call_greeks = {
    "Delta": call_delta(S, K, T, r, q, sigma),
    "Gamma": gamma(S, K, T, r, q, sigma),
    "Theta": call_theta(S, K, T, r, q, sigma),
    "Vega": vega(S, K, T, r, q, sigma),
    "Rho": call_rho(S, K, T, r, q, sigma)
}
put_greeks = {
    "Delta": put_delta(S, K, T, r, q, sigma),
    "Gamma": gamma(S, K, T, r, q, sigma),
    "Theta": put_theta(S, K, T, r, q, sigma),
    "Vega": vega(S, K, T, r, q, sigma),
    "Rho": put_rho(S, K, T, r, q, sigma)
}

# Calculate finite-difference Greeks
eps = 1e-4 

finite_delta_call = (call_price(S + eps, K, T, r, q, sigma) - call_price(S - eps, K, T, r, q, sigma)) / (2 * eps)
finite_gamma_call = (call_price(S + eps, K, T, r, q, sigma) - 2 * call_price(S, K, T, r, q, sigma) + call_price(S - eps, K, T, r, q, sigma)) / (eps ** 2)
finite_vega_call = (call_price(S, K, T, r, q, sigma + eps) - call_price(S, K, T, r, q, sigma - eps)) / (2 * eps)
finite_theta_call = (call_price(S, K, T - eps, r, q, sigma) - call_price(S, K, T + eps, r, q, sigma)) / (2 * eps)
finite_rho_call = (call_price(S, K, T, r + eps, q, sigma) - call_price(S, K, T, r - eps, q, sigma)) / (2 * eps)

# Finite-difference Greeks for put option
finite_delta_put = (put_price(S + eps, K, T, r, q, sigma) - put_price(S - eps, K, T, r, q, sigma)) / (2 * eps)
finite_gamma_put = (put_price(S + eps, K, T, r, q, sigma) - 2 * put_price(S, K, T, r, q, sigma) + put_price(S - eps, K, T, r, q, sigma)) / (eps ** 2)
finite_vega_put = (put_price(S, K, T, r, q, sigma + eps) - put_price(S, K, T, r, q, sigma - eps)) / (2 * eps)
finite_theta_put = (put_price(S, K, T - eps, r, q, sigma) - put_price(S, K, T + eps, r, q, sigma)) / (2 * eps)
finite_rho_put = (put_price(S, K, T, r + eps, q, sigma) - put_price(S, K, T, r - eps, q, sigma)) / (2 * eps)

# Calculate American Option Prices with and without Dividends
call_price_no_dividend = binomial_tree_american_option(S, K, T, r, q, sigma, n, option_type='call')
put_price_no_dividend = binomial_tree_american_option(S, K, T, r, q, sigma, n, option_type='put')
call_price_with_dividend = binomial_tree_american_option(S, K, T, r, q, sigma, n, option_type='call', dividend_date=dividend_date, dividend_amount=dividend)
put_price_with_dividend = binomial_tree_american_option(S, K, T, r, q, sigma, n, option_type='put', dividend_date=dividend_date, dividend_amount=dividend)

# Display Greeks results in a DataFrame
greeks_df = pd.DataFrame({
    "Greek": ["Delta", "Gamma", "Theta", "Vega", "Rho"],
    "Call Value": [
        call_greeks["Delta"], call_greeks["Gamma"], call_greeks["Theta"], 
        call_greeks["Vega"], call_greeks["Rho"]
    ],
    "Put Value": [
        put_greeks["Delta"], put_greeks["Gamma"], put_greeks["Theta"], 
        put_greeks["Vega"], put_greeks["Rho"]
    ]
})


finite_greeks_df = pd.DataFrame({
    "Greek": ["Delta", "Gamma", "Theta", "Vega", "Rho"],
    "Call Value": [finite_delta_call, finite_gamma_call, finite_theta_call, finite_vega_call, finite_rho_call],
    "Put Value": [finite_delta_put, finite_gamma_put, finite_theta_put, finite_vega_put, finite_rho_put]
})

comparison_df = pd.DataFrame({
    "Option Type": ["Call", "Put"],
    "Without Dividends": [call_price_no_dividend, put_price_no_dividend],
    "With Dividends": [call_price_with_dividend, put_price_with_dividend]
})

print("GBSM Greeks:")
print(greeks_df.to_string(index=False))
print("\nFinite Difference Greeks:")
print(finite_greeks_df.to_string(index=False))
print("\nDividend Comparison Analysis:")
print(comparison_df.to_string(index=False))

