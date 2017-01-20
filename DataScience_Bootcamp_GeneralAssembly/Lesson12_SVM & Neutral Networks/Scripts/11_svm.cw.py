#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 11: Support vector machines
'''

try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin

import numpy as np
import pandas as pd

from sklearn import preprocessing, svm, cross_validation as cv, grid_search

from sklearn.pipeline import Pipeline

# Select different Machine Learning Repository mirror if needed
MLR_MIRROR = 'http://archive.ics.uci.edu/ml/machine-learning-databases/'
#MLR_MIRROR = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/'

REDS_URL = urljoin(MLR_MIRROR, 'wine-quality/winequality-red.csv')

WHITES_URL = urljoin(MLR_MIRROR, 'wine-quality/winequality-white.csv')

# Read in the Wine Quality datasets
reds = pd.read_csv(REDS_URL, sep=';')
whites = pd.read_csv(WHITES_URL, sep=';')

# Add a new indicator variable for the type of wine
reds['red'] = 1
whites['red'] = 0

# Merge the two datasets
wines = pd.concat([reds, whites], axis=0)

# Prepare the data for use in scikit-learn
X = wines.drop(['quality', 'red'], axis=1)
y = wines.red.astype('int')

# Create a pipeline that scales the data and trains a support vector classifier
ssvc = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('svc', svm.SVC())
])

# Train a support vector classifier with linear (= no) kernel
ssvc.set_params(
    svc__kernel='linear'
)
ssvc.fit(X, y)

# Coefficients defining the separating hyperplane (NOT regression coefficients)
ssvc.named_steps['svc'].coef_

# Support vectors
ssvc.named_steps['svc'].n_support_
ssvc.named_steps['svc'].support_
ssvc.named_steps['svc'].support_vectors_

# Define stratified folds for cross-validation
kf = cv.StratifiedKFold(y, n_folds=10, shuffle=True)

# Compute average AUC across folds
aucs = cv.cross_val_score(ssvc, X, y, scoring='roc_auc', cv=kf)
np.mean(aucs)

# Train using the Radial Basis Function (RBF) kernel
ssvc.set_params(
    svc__kernel='rbf'
)
ssvc.fit(X, y)

# Compute average AUC across folds
aucs = cv.cross_val_score(ssvc, X, y, scoring='roc_auc', cv=kf)
np.mean(aucs)

# Determine ‘optimal’ kernel and value of C by cross-validation using AUC
# scoring
gs = grid_search.GridSearchCV(
    estimator=ssvc,
    param_grid={
        'svc__C': [1e-15, 0.0001, 0.001, 0.01, 0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf']
    },
    scoring='roc_auc',
    cv=kf
)
gs.fit(X, y)

gs.best_score_
gs.best_estimator_
gs.grid_scores_

