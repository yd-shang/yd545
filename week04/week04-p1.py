import numpy as np
import pandas as pd


P_t_minus_1 = 100  
sigma = 0.1  
n_simulations = 10000  

r_t = np.random.normal(0, sigma, n_simulations)

# Classical Brownian Motion
P_classical = P_t_minus_1 + r_t

# Arithmetic Return System
P_arithmetic = P_t_minus_1 * (1 + r_t)

# Log Return or Geometric Brownian Motion
P_log = P_t_minus_1 * np.exp(r_t)

# Calculate expected values and standard deviations (simulation results)
results = {
    'Model': ['Classical Brownian Motion', 'Arithmetic Return System', 'Geometric Brownian Motion'],
    'Mean': [np.mean(P_classical), np.mean(P_arithmetic), np.mean(P_log)],
    'Standard Deviation': [np.std(P_classical), np.std(P_arithmetic), np.std(P_log)]
}

results_df = pd.DataFrame(results)

# Expected theoretical values for each model
expected_values = {
    'Model': ['Classical Brownian Motion', 'Arithmetic Return System', 'Geometric Brownian Motion'],
    'Expected Mean': [P_t_minus_1, P_t_minus_1, P_t_minus_1 * np.exp(0)],
    'Expected Std Dev': [sigma, P_t_minus_1 * sigma, P_t_minus_1 * sigma]
}

expected_df = pd.DataFrame(expected_values)

combined_df = pd.merge(results_df, expected_df, on='Model')

print(combined_df)


