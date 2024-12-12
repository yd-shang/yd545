import numpy as np
import pandas as pd

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
    return_df = pd.DataFrame(p2, columns=vars)
    return return_df
