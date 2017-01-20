#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 15: Time series

* Aggregation
* Rolling and exponentially weighted averages
* Autocorrelation and stationarity
* AR(I)MA modelling
'''

import numpy as np
import pandas as pd

import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib qt

CYCLE_HIRES_URL = 'https://files.datapress.com/london/dataset/number-bicycle-hires/2016-05-17T09:24:57/tfl-daily-cycle-hires.xls'

# Load TfL Cycle Hire dataset
hires = pd.read_excel(CYCLE_HIRES_URL, sheetname='Data')

# Convert 'Day' to 'datetime' and set as index
hires.Day = pd.to_datetime(hires.Day, unit='D')
hires.set_index('Day', inplace=True)

# Extract first column (daily hires) and convert to 'float'
hires = hires.iloc[:,0].astype('float').to_frame('Hires')

# Plot time series
hires.Hires.plot()

'''
Aggregation
'''

# Resample at monthly resolution
monthly_hires = hires.Hires.resample('M').sum().to_frame()
monthly_hires.Hires.plot()

# Extract year, month, and weekday from index
hires['Year'] = hires.index.year
hires['Month'] = hires.index.month
hires['Weekday'] = hires.index.weekday

# Box plot of daily hires by year, month, and weekday
sns.boxplot(x='Year', y='Hires', data=hires)
sns.boxplot(x='Month', y='Hires', data=hires)
sns.boxplot(x='Weekday', y='Hires', data=hires)

'''
Rolling and exponentially weighted averages
'''

# Rolling average
hires.Hires.rolling(window=7, center=True).mean().plot()
hires.Hires.rolling(window=14, center=True).mean().plot()
hires.Hires.rolling(window=30, center=True).mean().plot()

# Exponentially weighted moving average
hires.Hires.ewm(alpha=0.01).mean().plot()
hires.Hires.ewm(alpha=0.02).mean().plot()
hires.Hires.ewm(alpha=0.05).mean().plot()

# Expanding average
hires.Hires.expanding().mean().plot()

'''
Autocorrelation and stationarity
'''

# Autocorrelation function (ACF)
plot_acf(hires.Hires, lags=30)

# Partial autocorrelation function (PACF)
plot_pacf(hires.Hires, lags=30)

# Rolling statistics
def plot_rolling_statistics(ts):
    rolling_stats = ts.rolling(window=30).agg(['mean', 'std'])
    plt.plot(ts, color='gray', label='Original')
    plt.plot(rolling_stats['mean'], color='red', label='Rolling mean')
    plt.plot(rolling_stats['std'], color='blue', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.show(block=False)

plot_rolling_statistics(hires.Hires)

# Augmented Dickey-Fuller test (small p-values suggest stationarity)
sm.tsa.adfuller(hires.Hires)

# Logarithmic transformation to stabilise variance
hires_diff = np.log10(hires.Hires)

plot_rolling_statistics(hires_diff)
plot_acf(hires_diff, lags=30)
plot_pacf(hires_diff, lags=30)
sm.tsa.adfuller(hires_diff)

# First-order differencing to induce stationarity
hires_diff = hires_diff - hires_diff.shift(1)
hires_diff.dropna(inplace=True)

plot_rolling_statistics(hires_diff)
plot_acf(hires_diff, lags=30)
plot_pacf(hires_diff, lags=30)
sm.tsa.adfuller(hires_diff)

'''
AR(I)MA modelling
'''

# Separate dataset into training and test sets
training = hires_diff[hires_diff.index < pd.to_datetime('2016-04-01')]
test = hires_diff[hires_diff.index >= pd.to_datetime('2016-04-01')]

# Fit ARMA(4, 2) model on the log-differences
model = sm.tsa.ARMA(training, (4, 2)).fit(maxiter=100)

model.summary()
model.summary2()

# Check ACF and PACF of residuals
plot_acf(model.resid, lags=30)
plot_pacf(model.resid, lags=30)

# Plot predictions and test set
model.plot_predict('2016-01-01', '2016-04-30')
test.plot()

# Alternatively, fit ARIMA(4, 1, 2) on the log-transformed counts
training = np.log10(hires.Hires[hires.index < pd.to_datetime('2016-04-01')])
test = np.log10(hires.Hires[hires.index >= pd.to_datetime('2016-04-01')])

model = sm.tsa.ARIMA(training, (4, 1, 2)).fit(maxiter=100)

model.summary()
model.summary2()

model.plot_predict('2016-01-01', '2016-04-30')
test.plot()

