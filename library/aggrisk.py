import pandas as pd
import numpy as np
from scipy.stats import norm

def calculate_var(series, alpha=0.05):
    """Calculate Value at Risk (VaR) for a given confidence level."""
    return -np.percentile(series, alpha * 100)

def calculate_es(series, alpha=0.05):
    """Calculate Expected Shortfall (ES) for a given confidence level."""
    var = calculate_var(series, alpha)
    return -series[series <= -var].mean()

def agg_risk(df, agg_level, current_agg_level=None):
    """
    Recursively aggregate risk metrics based on hierarchical levels.
    
    Parameters:
    - df: DataFrame containing the data for risk aggregation.
    - agg_level: List of column names for hierarchical aggregation.
    - current_agg_level: Internal recursive variable to track current level.
    
    Returns:
    - Aggregated risk metrics as a DataFrame.
    """
    if current_agg_level is None:
        current_agg_level = []
    
    if len(current_agg_level) < len(agg_level):
        # Determine the next level to aggregate
        next_level = agg_level[len(current_agg_level)]
        grouped = df.groupby(current_agg_level + [next_level])
        
        # Compute risk metrics for each group
        risk = grouped.apply(lambda group: pd.Series({
            'currentValue': group['value'].iloc[0],
            'VaR95': calculate_var(group['value'], alpha=0.05),
            'VaR99': calculate_var(group['value'], alpha=0.01),
            'ES95': calculate_es(group['value'], alpha=0.05),
            'ES99': calculate_es(group['value'], alpha=0.01),
            'std': group['value'].std(),
            'min': group['value'].min(),
            'max': group['value'].max(),
            'mean': group['value'].mean()
        })).reset_index()
        
        # Normalize VaR percentages
        risk['VaR95_Pct'] = risk['VaR95'] / risk['currentValue']
        risk['VaR99_Pct'] = risk['VaR99'] / risk['currentValue']
        risk['ES95_Pct'] = risk['ES95'] / risk['currentValue']
        risk['ES99_Pct'] = risk['ES99'] / risk['currentValue']
        
        # Recursively aggregate further
        aggregated = agg_risk(df, agg_level, current_agg_level + [next_level])
        aggregated[next_level] = 'Total'
        risk = pd.concat([risk, aggregated], ignore_index=True)
        return risk
    else:
        # Base case: Aggregate over all data
        total_metrics = pd.Series({
            'currentValue': df['value'].iloc[0],
            'VaR95': calculate_var(df['value'], alpha=0.05),
            'VaR99': calculate_var(df['value'], alpha=0.01),
            'ES95': calculate_es(df['value'], alpha=0.05),
            'ES99': calculate_es(df['value'], alpha=0.01),
            'std': df['value'].std(),
            'min': df['value'].min(),
            'max': df['value'].max(),
            'mean': df['value'].mean(),
        })
        total_metrics['VaR95_Pct'] = total_metrics['VaR95'] / total_metrics['currentValue']
        total_metrics['VaR99_Pct'] = total_metrics['VaR99'] / total_metrics['currentValue']
        total_metrics['ES95_Pct'] = total_metrics['ES95'] / total_metrics['currentValue']
        total_metrics['ES99_Pct'] = total_metrics['ES99'] / total_metrics['currentValue']
        
        return pd.DataFrame([total_metrics])

# Example Usage:
# Sample data
data = {
    'group1': ['A', 'A', 'A', 'B', 'B', 'B'],
    'group2': ['X', 'X', 'Y', 'Y', 'Z', 'Z'],
    'value': [100, 110, 95, 85, 80, 75]
}
df = pd.DataFrame(data)

# Aggregation levels
agg_level = ['group1', 'group2']

# Compute risk metrics
result = agg_risk(df, agg_level)
print(result)
