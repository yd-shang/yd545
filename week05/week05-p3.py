import numpy as np
import pandas as pd
from scipy.stats import norm, t

portfolio_df = pd.read_csv('portfolio.csv')
prices_df = pd.read_csv('DailyPrices.csv')

portfolio_stocks = portfolio_df['Stock'].unique()
dailyprice_stocks = prices_df.columns[1:] 
missing_stocks = [stock for stock in portfolio_stocks if stock not in dailyprice_stocks]
portfolio_df = portfolio_df[~portfolio_df['Stock'].isin(missing_stocks)]

# Arithmetic Returns
def return_calculate(prices: pd.DataFrame, method="DISCRETE", date_column="Date"):
    tickers = [col for col in prices.columns if col != date_column]
    p = prices[tickers].to_numpy()
    p2 = (p[1:, :] / p[:-1, :]) - 1.0  # Arithmetic returns
    returns_df = pd.DataFrame(p2, columns=tickers)
    returns_df.insert(0, date_column, prices[date_column].iloc[1:].reset_index(drop=True))
    returns_df[tickers] = returns_df[tickers] - returns_df[tickers].mean()  # Centering returns
    return returns_df

ars_returns_df = return_calculate(prices_df, method="DISCRETE")

portfolio_df['Distribution'] = portfolio_df['Portfolio'].apply(lambda x: 'T' if x in ['A', 'B'] else 'Normal')
for stock in portfolio_df["Stock"]:
    portfolio_df.loc[portfolio_df['Stock'] == stock, 'Starting Price'] = prices_df.iloc[-1][stock]

# Fit distributions to each stock
def fit_distributions(portfolio, returns):
    models = {}
    for stock in portfolio['Stock']:
        dist_type = portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0]
        if dist_type == 'Normal':
            mu, sigma = norm.fit(returns[stock])
            models[stock] = (mu, sigma)
        elif dist_type == 'T':
            nu, mu, sigma = t.fit(returns[stock])
            models[stock] = (nu, mu, sigma)
    return models

# Copula simulation with PCA-based Spearman correlation
def simulate_copula(portfolio, models, returns, nSim=10000):
    uniform = pd.DataFrame()
    standard_normal = pd.DataFrame()
    
    for stock in portfolio["Stock"]:
        dist_type = portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0]
        if dist_type == 'Normal':
            mu, sigma = models[stock]
            uniform[stock] = norm.cdf(returns[stock], loc=mu, scale=sigma)
            standard_normal[stock] = norm.ppf(uniform[stock])
        elif dist_type == 'T':
            nu, mu, sigma = models[stock]
            uniform[stock] = t.cdf(returns[stock], df=nu, loc=mu, scale=sigma)
            standard_normal[stock] = norm.ppf(uniform[stock])
    
    spearman_corr_matrix = standard_normal.corr(method='spearman')
    eigenvalues, eigenvectors = np.linalg.eigh(spearman_corr_matrix)
    B = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    rand_normals = np.random.normal(0.0, 1.0, size=(nSim, len(portfolio['Stock'])))
    simulated_returns = pd.DataFrame(rand_normals @ B.T, columns=portfolio['Stock'])
    
    return simulated_returns

# Generate simulated returns using the copula model
def generate_simulated_returns(portfolio, simulated_returns, models):
    for stock in portfolio['Stock']:
        dist_type = portfolio.loc[portfolio['Stock'] == stock, 'Distribution'].iloc[0]
        if dist_type == 'Normal':
            mu, sigma = models[stock]
            simulated_returns[stock] = norm.ppf(simulated_returns[stock], loc=mu, scale=sigma)
        elif dist_type == 'T':
            nu, mu, sigma = models[stock]
            simulated_returns[stock] = t.ppf(simulated_returns[stock], df=nu, loc=mu, scale=sigma)
    return simulated_returns

# Calculate VaR and ES for portfolio
def calculate_var_es(portfolio, simulated_returns, alpha=0.05):
    portfolio_value = (portfolio['Holding'] * portfolio['Starting Price']).sum()
    simulated_value = (1 + simulated_returns) * portfolio_value
    pnl = simulated_value - portfolio_value
    VaR = -np.percentile(pnl.sum(axis=1), alpha * 100) 
    ES = -pnl[pnl.sum(axis=1) <= -VaR].sum(axis=1).mean()
    VaR_pct = VaR / portfolio_value
    ES_pct = ES / portfolio_value
    return VaR, VaR_pct, ES, ES_pct

# Run the main process for each portfolio and total
portfolios = ["A", "B", "C"]
results = {}

for label in portfolios:
    each_portfolio = portfolio_df[portfolio_df["Portfolio"] == label]
    models = fit_distributions(each_portfolio, ars_returns_df)
    simulated_returns = simulate_copula(each_portfolio, models, ars_returns_df)
    simulated_returns = generate_simulated_returns(each_portfolio, simulated_returns, models)
    VaR, VaR_pct, ES, ES_pct = calculate_var_es(each_portfolio, simulated_returns)
    results[label] = (VaR, VaR_pct, ES, ES_pct)
    print(f"Portfolio {label} VaR in $: {VaR:.2f}")
    print(f"Portfolio {label} ES in $: {ES:.2f}")

# Calculate total portfolio VaR and ES
models_total = fit_distributions(portfolio_df, ars_returns_df)
simulated_returns_total = simulate_copula(portfolio_df, models_total, ars_returns_df)
simulated_returns_total = generate_simulated_returns(portfolio_df, simulated_returns_total, models_total)
total_VaR, total_VaR_pct, total_ES, total_ES_pct = calculate_var_es(portfolio_df, simulated_returns_total)

print(f"Total Portfolio VaR in $: {total_VaR:.2f}")
print(f"Total Portfolio ES in $: {total_ES:.2f}")

results, (total_VaR, total_VaR_pct, total_ES, total_ES_pct)
