#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 6: Decision trees and random forests
'''

import numpy as np
import pandas as pd

from sklearn import cross_validation as cv, tree, ensemble

REDS_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

WHITES_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

# Read in the Wine Quality datasets
reds = pd.read_csv(REDS_URL, sep=';')
whites = pd.read_csv(WHITES_URL, sep=';')

# Add a new indicator variable for the type of wine
reds['red'] = 1
whites['red'] = 0

# Merge the two datasets
wines = pd.concat([reds, whites], axis=0)

# Prepare the data for use in scikit-learn
X = wines.drop('quality', axis=1)
y = wines.quality

# Train a decision tree by limiting the depth to 3, and the minimum number of
# samples per leaf to 50

# Export the tree for plotting

# Define folds for cross-validation

# Compute average MSE across folds

# Train a random forest with 20 decision trees

# Investigate importances of predictors

# Evaluate performance through cross-validation

# What happens when you increase the number of trees to 50?

