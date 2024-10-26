import pandas as pd
import numpy as np
from scipy.stats import norm, t

file_path = '/Users/shangyudi/Desktop/Duke/2024fall/quantitative risk management/week05/problem1.csv'
data = pd.read_csv(file_path)

confidence_level = 0.95

n = len(data)

sorted_returns = np.sort(data['x'])

# (a) VaR and ES using normal distribution with exponentially weighted variance (lambda=0.97)
lambda_ewma = 0.97

weights = np.array([lambda_ewma**i for i in range(n)][::-1])
weights /= weights.sum()  

ewma_mean = np.sum(weights * data['x'])
ewma_variance = np.sum(weights * (data['x'] - ewma_mean)**2)
ewma_std = np.sqrt(ewma_variance)

# VaR
z_score = norm.ppf(1 - confidence_level)
var_ewma_normal = ewma_mean + z_score * ewma_std

# ES 
es_ewma_normal = -ewma_mean + (1 / (1-confidence_level)) * norm.pdf(norm.ppf(1 - confidence_level)) * ewma_std

# (b) VaR and ES using MLE-fitted T-distribution
params = t.fit(data['x'])  # Fitting the T-distribution
df_t, loc_t, scale_t = params

# VaR
var_mle_t = loc_t + scale_t * t.ppf(1 - confidence_level, df_t)

# ES
es_mle_t = -loc_t + (1 / (1-confidence_level)) * (t.pdf(t.ppf(1 - confidence_level, df_t), df_t)*(df_t + t.ppf(1 - confidence_level, df_t)**2) / (df_t - 1)) * scale_t

# (c) Historical simulation
# VaR
var_historical = np.percentile(sorted_returns, (1 - confidence_level) * 100)

# ES
es_historical = -sorted_returns[sorted_returns <= var_historical].mean()

# Print the results
results = {
    "VaR Normal (EWMA)": -var_ewma_normal,
    "ES Normal (EWMA)": es_ewma_normal,
    "VaR MLE T-distribution": -var_mle_t,
    "ES MLE T-distribution": es_mle_t,
    "VaR Historical": -var_historical,
    "ES Historical": es_historical
}

for key, value in results.items():
    print(f"{key}: {value:.6f}")
