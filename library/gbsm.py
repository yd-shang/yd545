import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar

class OptionResult:
    """
    A class to encapsulate the result of the GBSM calculation, including price, Greeks, intrinsic value, and time value.
    """
    def __init__(self, price, intrinsic_value=None, time_value=None, delta=None, gamma=None, vega=None, theta=None, cRho=None):
        self.price = price
        self.intrinsic_value = intrinsic_value
        self.time_value = time_value
        self.delta = delta
        self.gamma = gamma
        self.vega = vega
        self.theta = theta
        self.cRho = cRho

def gbsm(call, underlying, strike, ttm, rf, b, ivol, include_greeks=False):
    """
    Generalized Black-Scholes-Merton (GBSM) option pricing model with optional Greek calculations.

    Parameters:
        call (bool): True for call option, False for put option.
        underlying (float): Current price of the underlying asset.
        strike (float): Strike price of the option.
        ttm (float): Time to maturity (in years).
        rf (float): Risk-free interest rate (as a decimal, e.g., 0.05 for 5%).
        b (float): Cost-of-carry term (varies depending on the model used).
        ivol (float): Implied volatility (as a decimal, e.g., 0.2 for 20%).
        include_greeks (bool): Whether to calculate Greeks (default is False).

    Returns:
        OptionResult: Object containing the price, Greeks, intrinsic value, and time value.
    """
    # Calculate d1 and d2
    d1 = (np.log(underlying / strike) + (b + ivol**2 / 2) * ttm) / (ivol * np.sqrt(ttm))
    d2 = d1 - ivol * np.sqrt(ttm)

    # Calculate the option price
    if call:
        price = (underlying * np.exp((b - rf) * ttm) * norm.cdf(d1) - 
                 strike * np.exp(-rf * ttm) * norm.cdf(d2))
    else:
        price = (strike * np.exp(-rf * ttm) * norm.cdf(-d2) - 
                 underlying * np.exp((b - rf) * ttm) * norm.cdf(-d1))

    # Calculate intrinsic value
    if call:
        intrinsic_value = max(0, underlying - strike)
    else:
        intrinsic_value = max(0, strike - underlying)

    # Calculate time value
    time_value = price - intrinsic_value

    if not include_greeks:
        return OptionResult(price=price, intrinsic_value=intrinsic_value, time_value=time_value)

    # Calculate Greeks
    delta = np.exp((b - rf) * ttm) * norm.cdf(d1) if call else -np.exp((b - rf) * ttm) * norm.cdf(-d1)
    gamma = np.exp((b - rf) * ttm) * norm.pdf(d1) / (underlying * ivol * np.sqrt(ttm))
    vega = underlying * np.exp((b - rf) * ttm) * norm.pdf(d1) * np.sqrt(ttm)
    theta = (-underlying * np.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * np.sqrt(ttm)) -
             (b - rf) * underlying * np.exp((b - rf) * ttm) * norm.cdf(d1) if call else
             -underlying * np.exp((b - rf) * ttm) * norm.pdf(-d1) * ivol / (2 * np.sqrt(ttm)) +
             rf * strike * np.exp(-rf * ttm) * norm.cdf(d2) if call else
             rf * strike * np.exp(-rf * ttm) * norm.cdf(-d2))
    cRho = ttm * underlying * np.exp((b - rf) * ttm) * (norm.cdf(d1) if call else norm.cdf(-d1))

    return OptionResult(price=price, intrinsic_value=intrinsic_value, time_value=time_value,
                        delta=delta, gamma=gamma, vega=vega, theta=theta, cRho=cRho)

def calculate_implied_volatility(option_type, underlying, strike, ttm, rf, b, market_price, initial_guess=0.2):
    """
    Calculate the implied volatility for a given option using the GBSM model.

    Parameters:
        option_type (str): "Call" for call option, "Put" for put option.
        underlying (float): Current price of the underlying asset.
        strike (float): Strike price of the option.
        ttm (float): Time to maturity (in years).
        rf (float): Risk-free interest rate (as a decimal).
        b (float): Cost-of-carry term (depends on the model).
        market_price (float): Observed market price of the option.
        initial_guess (float): Initial guess for the implied volatility (default is 0.2).

    Returns:
        float: Implied volatility (as a decimal, e.g., 0.2 for 20%).
    """
    def objective(ivol):
        # Use the GBSM function to calculate the theoretical price
        theoretical_price = gbsm(option_type == "Call", underlying, strike, ttm, rf, b, ivol).price
        return theoretical_price - market_price

    # Use root_scalar to solve for implied volatility
    result = root_scalar(objective, bracket=[1e-6, 5], method='brentq')  # Bracket specifies the search range
    if result.converged:
        return result.root
    else:
        raise ValueError("Failed to converge to a solution for implied volatility.")

