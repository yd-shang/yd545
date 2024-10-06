import numpy as np
import pandas as pd
from scipy.stats import norm, t
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

def return_calculate(prices, method="DISCRETE", date_column="date"):
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame")
    
    vars = prices.columns.difference([date_column])
    p = prices[vars].to_numpy()
    n = p.shape[0]
    m = p.shape[1]
    
    # Prepare the return matrix
    p2 = np.zeros((n - 1, m))
    
    # Calculate returns based on the selected method
    for i in range(n - 1):
        for j in range(m):
            p2[i, j] = p[i + 1, j] / p[i, j]
    
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in ('LOG', 'DISCRETE')")
    
    # Return a DataFrame with the same structure
    dates = prices[date_column].iloc[1:].reset_index(drop=True)
    return_df = pd.DataFrame(p2, columns=vars, index=dates)
    return return_df

file_path = 'DailyPrices.csv' 
daily_prices = pd.read_csv(file_path)

# Calculate discrete returns for all stocks
arithmetic_returns = return_calculate(daily_prices, method='DISCRETE', date_column='Date')

# Remove the mean from META returns
meta_returns = arithmetic_returns['META']
meta_returns_adjusted = meta_returns - meta_returns.mean()

# Update META in the return DataFrame
arithmetic_returns['META'] = meta_returns_adjusted

# VaR calculation functions (unchanged)
def calculate_var_normal(returns, confidence_level=0.95):
    mean_return = returns.mean()
    std_dev = returns.std()
    z_score = norm.ppf(1 - confidence_level)
    var = mean_return + z_score * std_dev
    return -var 

def calculate_var_ewma(returns, confidence_level=0.95, lambda_=0.94):
    weights = [(1 - lambda_) * lambda_**i for i in range(len(returns))]
    weights = np.array(weights[::-1])
    weighted_variance = np.sum(weights * (returns - returns.mean())**2)
    weighted_std_dev = np.sqrt(weighted_variance)
    z_score = norm.ppf(1 - confidence_level)
    var = z_score * weighted_std_dev
    return -var  

def calculate_var_mle_t(returns, confidence_level=0.95):
    df, loc, scale = t.fit(returns)
    var = t.ppf(1 - confidence_level, df, loc=loc, scale=scale)
    return -var 

def calculate_var_ar1(returns, confidence_level=0.95):
    model = AutoReg(returns, lags=1).fit()
    predictions = model.forecast(steps=1)
    residual_std = np.std(model.resid)
    z_score = norm.ppf(1 - confidence_level)
    var = predictions.iloc[0] + z_score * residual_std  
    return -var

def calculate_var_historical(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index]
    return -var 

# Calculate VaR for META using all 5 methods
meta_var_normal = calculate_var_normal(meta_returns_adjusted, confidence_level=0.95)
meta_var_ewma = calculate_var_ewma(meta_returns_adjusted, confidence_level=0.95, lambda_=0.94)
meta_var_mle_t = calculate_var_mle_t(meta_returns_adjusted, confidence_level=0.95)
meta_var_ar1 = calculate_var_ar1(meta_returns_adjusted, confidence_level=0.95)
meta_var_historical = calculate_var_historical(meta_returns_adjusted, confidence_level=0.95)

# Print VaR results
print("VaR Calculations:")
print(f"Normal VaR: {meta_var_normal}")
print(f"EWMA VaR: {meta_var_ewma}")
print(f"MLE T-Distribution VaR: {meta_var_mle_t}")
print(f"AR(1) VaR: {meta_var_ar1}")
print(f"Historical VaR: {meta_var_historical}")


