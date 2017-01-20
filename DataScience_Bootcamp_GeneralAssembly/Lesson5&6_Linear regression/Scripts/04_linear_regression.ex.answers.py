#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 4: Linear regression using StatsModels
Model answers
'''

import os

import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.gofplots as smg

from pandas.tools.plotting import autocorrelation_plot

%matplotlib qt

sns.set(rc={
    'figure.figsize': (8, 6),
    'font.size': 14
})

# Read in the EU referendum data from the 'Web scraping' code walk-through
brexit = pd.read_csv(os.path.join('datasets', 'brexit.csv'))

# Change the data type of 'date' to `datetime`
brexit.date = pd.to_datetime(brexit.date)

# Create a new variable 't' representing the years between 2015-01-01 and 'date'
# (Hint: convert to `timedelta64[D]` first, then divide by 365.25)
brexit['t'] = (brexit.date - pd.to_datetime('2015-01-01')).astype('timedelta64[D]') / 365.25

# Build a regression model for 'stay' versus 't'
model1 = smf.ols('stay ~ t', data=brexit).fit()

# Examine the model output
model1.summary()
model1.summary2()

# Produce the following diagnostic plots:

# * Predicted versus observed
sns.jointplot(brexit.stay, model1.fittedvalues)

# * Residuals versus predicted
sns.jointplot(model1.fittedvalues, model1.resid)

# * Residuals versus 't'
sns.pointplot(brexit.t, model1.resid, join=False)

# * Autocorrelation plot
autocorrelation_plot(model1.resid)

# * Normal Q-Q plot for (Studentised) residuals
st_resid = model1.get_influence().get_resid_studentized_external()
qq = smg.qqplot(st_resid)
smg.qqline(qq.gca(), '45')

# BONUS: Build a second regression model for 'stay' versus 't' and 'pollster',
#        and re-run all of the above
model2 = smf.ols('stay ~ t + pollster', data=brexit).fit()

