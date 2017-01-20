#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 5: Logistic regression using StatsModels
'''

import os

import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

%matplotlib qt

BANKNOTES_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'

var_names = ['wavelet_var', 'wavelet_skew', 'wavelet_kurt', 'entropy', 'forged']

# Read in the Banknote Authentication dataset

# Explore data visually

# Build a logistic regression model without predictors

# Build one logistic regression model for each predictor

# Select the ‘best’ predictor based on the AIC, and build three models including
# this variable and each of the remaining three
# Rule of thumb: ΔAIC < 2  = No difference, prefer model with less predictors
#                     < 6  = Model with lower AIC is preferred (assuming large N)
#                     < 10 = Model with lower AIC is preferred (assuming small N)
#                     ≥ 10 = Model with lower AIC is strongly preferred (always)

# Repeat, building two models including the two ‘most predictive’ variables and
# each of the remaining predictors

# Finally, build the last model including all predictors

# Print out and interpret the coefficients of the ‘most predictive’ model

