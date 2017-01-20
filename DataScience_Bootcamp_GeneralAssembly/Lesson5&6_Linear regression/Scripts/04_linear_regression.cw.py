#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 4: Linear regression using StatsModels

* Design matrix
* Linear regression
* Diagnostics
* Transformations
'''

import os

import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.gofplots as smg
import statsmodels.stats.diagnostic as smd
import statsmodels.stats.stattools as sms

from pandas.tools.plotting import autocorrelation_plot

from patsy import dmatrices

from scipy.stats import anderson

%matplotlib qt

sns.set(rc={
    'figure.figsize': (8, 6),
    'font.size': 14
})

# Read in the National Child Development Study data
ncds = pd.read_csv(os.path.join('datasets', 'ncds.csv'), index_col=0)

# Define predictors and response
predictors = ['n920', 'n923', 'n926', 'n851', 'n852']
response = 'earngsn'

all_vars = predictors + [response]

# Recode missing values
ncds.where(ncds[all_vars] > 0, inplace=True)

# Remove missing values
ncds.dropna(subset=all_vars, inplace=True)

# Explore data visually
sns.pairplot(ncds, x_vars=predictors, y_vars=response, kind='reg')
sns.lmplot(x='n923', y='n926', data=ncds)

# Correlation matrix (ranges from -1 to 1)
ncds[all_vars].corr()

# Visualise correlation matrix using a heatmap
sns.heatmap(ncds[all_vars].corr())

'''
Design matrix
'''

# Patsy can be used to construct the design matrices from a formula
y, X = dmatrices('earngsn ~ n920 + n923 + n926', data=ncds)

# Dummies for categorical variables are automatically constructed by 'C'
y, X = dmatrices('n920 ~ C(n851, Treatment(reference=3))', data=ncds)

'''
Linear regression
'''

# 'Ordinary Least Squares'
model1 = smf.ols('earngsn ~ n920 + n923 + n926', data=ncds).fit()
model1.summary()
model1.summary2()

model1.params
model1.pvalues
model1.conf_int()
model1.rsquared

'''
Diagnostics
'''

# Predicted versus observed
sns.jointplot(ncds[response], model1.fittedvalues)

# Residuals versus predicted
sns.jointplot(model1.fittedvalues, model1.resid)

# Residuals versus 'n920' (predictor)
sns.jointplot(ncds.n920, model1.resid)

# Residuals versus row number
sns.pointplot(ncds.index, model1.resid, join=False)

# Autocorrelation plot (most useful for time series)
autocorrelation_plot(model1.resid)

# Durbin-Watson test for autocorrelation at lag 1
sms.durbin_watson(model1.resid)

# Studentised residuals
st_resid = model1.get_influence().get_resid_studentized_external()

# Studentised residuals versus row number
sns.pointplot(ncds.index, st_resid, join=False)

# Normal Q-Q plot for residuals
qq = smg.qqplot(st_resid)
smg.qqline(qq.gca(), '45')

# Anderson-Darling test for normality
anderson(model1.resid)

'''
Transformations
'''

# Create new feature to represent 'parental interest'
ncds['parental_interest'] = ((ncds.n851 <= 2) & (ncds.n852 <= 2)).astype('int')

# Apply logarithmic transformation to 'earngsn'
model2 = smf.ols('np.log(earngsn) ~ parental_interest', data=ncds).fit()

model2.summary()
model2.summary2()

