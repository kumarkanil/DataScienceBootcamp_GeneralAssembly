#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 4: Linear regression using StatsModels
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

# Change the data type of 'date' to `datetime`

# Create a new variable 't' representing the years between 2015-01-01 and 'date'
# (Hint: convert to `timedelta64[D]` first, then divide by 365.25)

# Build a regression model for 'stay' versus 't'

# Examine the model output

# Produce the following diagnostic plots:

# * Predicted versus observed

# * Residuals versus predicted

# * Residuals versus 't'

# * Autocorrelation plot

# * Normal Q-Q plot for (Studentised) residuals

# BONUS: Build a second regression model for 'stay' versus 't' and 'pollster',
#        and re-run all of the above

