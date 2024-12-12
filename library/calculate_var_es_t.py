import numpy as np
from scipy.stats import t

def calculate_var_es_t(t_params, confidence_level=0.05):
    df, loc, scale = t_params
    # Calculate VaR (5th percentile)
    var = -t.ppf(confidence_level, df, loc, scale)  # Negate to make positive
    # Calculate Expected Shortfall (ES)
    es = -t.expect(lambda x: x, args=(df,), loc=loc, scale=scale, lb=-np.inf, ub=-var) / confidence_level  # Negate ES
    
    return var, es