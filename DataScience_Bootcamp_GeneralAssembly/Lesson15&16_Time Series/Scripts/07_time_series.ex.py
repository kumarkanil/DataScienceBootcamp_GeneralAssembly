#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 7: Time series
'''

import numpy as np
import pandas as pd

from sklearn import preprocessing, svm, cross_validation as cv, grid_search
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

%matplotlib qt

CYCLE_HIRES_URL = 'https://files.datapress.com/london/dataset/number-bicycle-hires/2016-05-17T09:24:57/tfl-daily-cycle-hires.xls'

# Load TfL Cycle Hire dataset
hires = pd.read_excel(CYCLE_HIRES_URL, sheetname='Data')

# Convert 'Day' to 'datetime' and set as index
hires.Day = pd.to_datetime(hires.Day, unit='D')
hires.set_index('Day', inplace=True)

# Extract first column (daily hires) and convert to 'float'
hires = hires.iloc[:,0].astype('float').to_frame('Hires')

# Apply logarithmic transformation

# Create 7 new variables representing the lagged time series at lag = 1, ..., 7

# Create 2 new variables representing the smoothed time series
# (rolling averages at 7 and 30 days)

# Drop missing values

# Define cross-validation split by leaving out 2016 as test set
split = cv.PredefinedSplit(test_fold=(hires.index.year == 2016) - 1)

# Create a pipeline that scales the data and trains a support vector regression
# model

# Fit the model

# Compute MSE for split

# Determine ‘optimal’ kernel and value of C by cross-validation

# Plot original time series and prediction from January 2015 onwards

