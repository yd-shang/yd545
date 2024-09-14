import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('problem3.csv')

print(data.head())

plt.figure(figsize=(10, 6))
plt.plot(data['x']) 
plt.title('Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value(x)')
plt.grid(True)
plt.show()

plot_acf(data['x'], lags=30)
plt.show()

plot_pacf(data['x'], lags=30)
plt.show()

from statsmodels.tsa.ar_model import AutoReg

# AR(1)
ar1_model = AutoReg(data['x'], lags=1).fit()
print(f'AR(1) AIC: {ar1_model.aic}')

# AR(2)
ar2_model = AutoReg(data['x'], lags=2).fit()
print(f'AR(2) AIC: {ar2_model.aic}')

# AR(3)
ar3_model = AutoReg(data['x'], lags=3).fit()
print(f'AR(3) AIC: {ar3_model.aic}')

from statsmodels.tsa.arima.model import ARIMA

# MA(1)
ma1_model = ARIMA(data['x'], order=(0, 0, 1)).fit()
print(f'MA(1) AIC: {ma1_model.aic}')

# MA(2)
ma2_model = ARIMA(data['x'], order=(0, 0, 2)).fit()
print(f'MA(2) AIC: {ma2_model.aic}')

# MA(3)
ma3_model = ARIMA(data['x'], order=(0, 0, 3)).fit()
print(f'MA(3) AIC: {ma3_model.aic}')

# campare AIC
aic_values = {
    'AR(1)': ar1_model.aic,
    'AR(2)': ar2_model.aic,
    'AR(3)': ar3_model.aic,
    'MA(1)': ma1_model.aic,
    'MA(2)': ma2_model.aic,
    'MA(3)': ma3_model.aic
}

best_model = min(aic_values, key=aic_values.get)
print(f"The best fitting model is: {best_model}")


