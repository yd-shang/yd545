import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

data = pd.read_csv('problem1.csv')
x = data['x']

#problem1-1
#Manually calculate
mean_manual = np.mean(x)
variance_manual = np.var(x, ddof=1)
skewness_manual = np.mean(((x - mean_manual) / np.std(x, ddof=1))**3)
kurtosis_manual = np.mean(((x - mean_manual) / np.std(x, ddof=1))**4) - 3

#problem1-2
#Use scipy package
mean_scipy = np.mean(x)
variance_scipy = np.var(x, ddof=1)
skewness_scipy = skew(x)
kurtosis_scipy = kurtosis(x, fisher=True)

print(f"Manual Mean: {mean_manual}, Scipy Mean: {mean_scipy}")
print(f"Manual Variance: {variance_manual}, Scipy Variance: {variance_scipy}")
print(f"Manual Skewness: {skewness_manual}, Scipy Skewness: {skewness_scipy}")
print(f"Manual Kurtosis: {kurtosis_manual}, Scipy Kurtosis: {kurtosis_scipy}")


