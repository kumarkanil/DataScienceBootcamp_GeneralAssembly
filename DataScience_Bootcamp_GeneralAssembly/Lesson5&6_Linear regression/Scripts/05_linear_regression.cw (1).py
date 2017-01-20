#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 5: Linear regression using scikit-learn

* Linear regression
* Cross-validation
* Regularisation:
  - L1 penalty (LASSO)
  - L2 penalty (Ridge)
  - Elastic Net
'''

import os

import numpy as np
import pandas as pd

from sklearn import linear_model as lm
from sklearn import metrics
from sklearn import cross_validation as cv
from sklearn import grid_search

import matplotlib.pyplot as plt

%matplotlib qt

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

'''
Linear regression
'''

# Prepare data
test_scores = ['n920', 'n923', 'n926']
X = ncds[test_scores]
y = ncds[response]

# Fit the model
model1 = lm.LinearRegression()
model1.fit(X, y)

# Print regression coefficients
model1.intercept_
model1.coef_

# Compute the prediction for [45, 20, 20] manually
new_sample = [45, 20, 20]
model1.intercept_ + sum(model1.coef_ * new_sample)

# Compute the prediction for [45, 20, 20] using `predict`
new_sample = np.matrix([45, 20, 20])
model1.predict(new_sample)

# Create dummy variables for n851 and n852
# (As an alternative to `dmatrices`; see previous code walk-through)
n851_dummies = pd.get_dummies(ncds.n851.astype('int'), prefix='n851')
n852_dummies = pd.get_dummies(ncds.n852.astype('int'), prefix='n852')

# 'Some interest' used as reference: drop the third column
n851_dummies.drop(n851_dummies.columns[2], axis=1, inplace=True)
n852_dummies.drop(n852_dummies.columns[2], axis=1, inplace=True)

# Concatenate the original DataFrame with the dummies
ncds = pd.concat([ncds, n851_dummies, n852_dummies], axis=1)

# Include dummy variables in the model
predictors = test_scores +\
             n851_dummies.columns.tolist() +\
             n852_dummies.columns.tolist()

# Fit the model
model2 = lm.LinearRegression()
model2.fit(ncds[predictors], ncds[response])

# Print regression coefficients
model2.intercept_
model2.coef_

# Pair variable names and coefficients
list(zip(predictors, model2.coef_))

'''
Cross-validation
'''

# Define 10 folds
kf = cv.KFold(len(ncds), n_folds=10, shuffle=True)

# Compute MSEs (one per fold)
mses = []
for train_idx, test_idx in kf:
    model = lm.LinearRegression()
    model.fit(ncds[predictors].iloc[train_idx], ncds[response].iloc[train_idx])
    mses.append(metrics.mean_squared_error(
        ncds[response].iloc[test_idx],
        model.predict(ncds[predictors].iloc[test_idx])
    ))

# Compute average MSE across folds
np.mean(mses)

# Alternatively, using `cross_val_score`…
mses = cv.cross_val_score(lm.LinearRegression(),\
                          ncds[predictors], ncds[response],\
                          scoring='mean_squared_error',\
                          cv=kf)

# Fix the sign and compute average MSE
# (see https://github.com/scikit-learn/scikit-learn/issues/2439)
(-mses).mean()

# Compare to the MSE for the model estimated on the entire dataset
metrics.mean_squared_error(ncds[response], model2.predict(ncds[predictors]))

'''
Regularisation: L1 penalty (LASSO)
'''

# Fit the model (by default, alpha = 1)
model2_lasso = lm.Lasso().fit(ncds[predictors], ncds[response])

# Print regression coefficients
model2_lasso.intercept_
list(zip(predictors, model2_lasso.coef_))

# Compute MSE
metrics.mean_squared_error(ncds[response], model2_lasso.predict(ncds[predictors]))

# Determine ‘optimal’ value of alpha using grid search with cross-validation
gs = grid_search.GridSearchCV(estimator=lm.Lasso(),\
                              param_grid={'alpha': np.logspace(-10, 10, 21)},\
                              scoring='mean_squared_error',\
                              cv=kf)

gs.fit(ncds[predictors], ncds[response])

# ‘Best’ MSE (fixing the sign again)
-gs.best_score_

# ‘Best’ model (includes value of alpha)
gs.best_estimator_

# All grid configurations and corresponding performances
gs.grid_scores_

# Alternatively, using `LassoCV`…
model2_lasso = lm.LassoCV(cv=kf).fit(ncds[predictors], ncds[response])

# Plot (average) MSE across folds versus alpha
nlog_alphas = -np.log10(model2_lasso.alphas_)
plt.figure()
plt.plot(nlog_alphas, model2_lasso.mse_path_, ':')
plt.plot(nlog_alphas, model2_lasso.mse_path_.mean(axis=-1), linewidth=2, color='k')
plt.axvline(-np.log10(model2_lasso.alpha_), linestyle='--', color='k')

'''
Regularisation: L2 penalty (Ridge)
'''

# Fit the model (by default, alpha = 1)
model2_ridge = lm.Ridge().fit(ncds[predictors], ncds[response])

# Print regression coefficients
model2_ridge.intercept_
list(zip(predictors, model2_ridge.coef_))

# Compute MSE
metrics.mean_squared_error(ncds[response], model2_ridge.predict(ncds[predictors]))

# Determine ‘optimal’ value of alpha using grid search with cross-validation
gs = grid_search.GridSearchCV(estimator=lm.Ridge(),\
                              param_grid={'alpha': np.logspace(-10, 10, 21)},\
                              scoring='mean_squared_error',\
                              cv=kf)

gs.fit(ncds[predictors], ncds[response])

# ‘Best’ MSE (fixing the sign again)
-gs.best_score_

# ‘Best’ model (includes value of alpha)
gs.best_estimator_

# All grid configurations and corresponding performances
gs.grid_scores_

