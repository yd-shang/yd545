#p2-1
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm

data = pd.read_csv('problem2.csv')
x = data['x'].values
y = data['y'].values

X_ols = sm.add_constant(x) 
ols_model = sm.OLS(y, X_ols).fit()
residuals = y - ols_model.predict(X_ols)
sigma_ols = np.sqrt(np.sum(residuals**2) / (len(y) - 2))
print(f"OLS coefficients: {ols_model.params}")
print(f'Standard Deviation (OLS): {sigma_ols}')

def negative_log_likelihood(params):
    beta0, beta1, sigma = params
    if sigma <= 0:
        return np.inf 
    predicted = beta0 + beta1 * x
    residuals = y - predicted
    n = len(y)
    nll = 0.5 * n * np.log(2 * np.pi * sigma**2) + np.sum(residuals**2) / (2 * sigma**2)
    return nll

#OLS
initial_params = [ols_model.params[0], ols_model.params[1], np.std(y - ols_model.predict(X_ols))]

result = minimize(negative_log_likelihood, initial_params, method='L-BFGS-B', bounds=[(None, None), (None, None), (1e-5, None)])

beta0_mle, beta1_mle, sigma_mle = result.x
print(f"MLE Estimated Intercept (beta0): {beta0_mle}")
print(f"MLE Estimated Slope (beta1): {beta1_mle}")
print(f"Standard Deviation of MLE: {sigma_mle}")

#p2-2
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln 

# Define the negative log-likelihood function for the t-distribution
def negative_log_likelihood_t(params):
    beta0, beta1, sigma, nu = params[0], params[1], params[2], params[3]
    if sigma <= 0 or nu <= 2:
        return np.inf  
    predicted = beta0 + beta1 * x
    residuals = y - predicted
    n = len(y)
    
    term1 = gammaln((nu + 1) / 2) - gammaln(nu / 2)
    term2 = -0.5 * np.log(nu * np.pi * sigma**2)
    term3 = -(nu + 1) / 2 * np.log(1 + (residuals / sigma)**2 / nu)
    
    nll = -np.sum(term1 + term2 + term3)
    return nll

initial_params = [ols_model.params[0], ols_model.params[1], np.std(y - ols_model.predict(X_ols)), 5]

result = minimize(negative_log_likelihood_t, initial_params, method='L-BFGS-B', bounds=[(None, None), (None, None), (1e-5, None), (2, None)])

beta0_mle_t, beta1_mle_t, sigma_mle_t, nu_mle_t = result.x

print(f"MLE Estimated Intercept (beta0) with t-distribution: {beta0_mle_t}")
print(f"MLE Estimated Slope (beta1) with t-distribution: {beta1_mle_t}")
print(f"MLE Estimated Standard Deviation (sigma) with t-distribution: {sigma_mle_t}")
print(f"MLE Estimated Degrees of Freedom (nu): {nu_mle_t}")


#p2-3
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

data_x = pd.read_csv('problem2_x.csv')
x1 = data_x['x1']
x2 = data_x['x2']

# add intercept
X = sm.add_constant(data_x)

#OLS 
y = np.random.randn(len(x1)) 
model = sm.OLS(y, X).fit()

print(model.summary())

y_pred = model.predict(X)

residuals = y - y_pred

# plot
plt.figure(figsize=(10, 6))
plt.scatter(x1, x2, label='x1 vs x2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter plot of x1 vs x2')
plt.grid(True)
plt.show()

# Plot the conditional distribution of x2 given x1
plt.figure(figsize=(10, 6))
for i in range(0, len(x1), int(len(x1) / 10)): 
plt.xlabel('x1 (Fixed values)')
plt.ylabel('x2 (Conditional Distribution)')
plt.title('Conditional Distribution of x2 given x1')
plt.grid(True)
plt.show()
