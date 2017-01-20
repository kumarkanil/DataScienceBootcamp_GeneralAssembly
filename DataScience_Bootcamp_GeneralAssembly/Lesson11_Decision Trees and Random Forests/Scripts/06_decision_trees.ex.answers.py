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
tree1 = tree.DecisionTreeRegressor(max_depth=2, min_samples_leaf=50)
tree1.fit(X, y)

# Export the tree for plotting
tree.export_graphviz(tree1, 'tree1.dot', feature_names=X.columns)

# Define folds for cross-validation
kf = cv.KFold(n_folds=10, shuffle=True)

# Compute average MSE across folds
mses = cv.cross_val_score(
    tree.DecisionTreeRegressor(max_depth=2, min_samples_leaf=50),\
    X, y, scoring='mean_squared_error', cv=kf)
np.mean(-mses)

# Train a random forest with 20 decision trees
rf1 = ensemble.RandomForestRegressor(n_estimators=20)
rf1.fit(X, y)

# Investigate importances of predictors
rf1.feature_importances_

# Evaluate performance through cross-validation
mses = cv.cross_val_score(ensemble.RandomForestRegressor(n_estimators=20),\
                          X, y, scoring='mean_squared_error', cv=kf)
np.mean(-mses)

# What happens when you increase the number of trees to 50?
mses = cv.cross_val_score(ensemble.RandomForestRegressor(n_estimators=50),\
                          X, y, scoring='mean_squared_error', cv=kf)
np.mean(-mses)

