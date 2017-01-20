#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 10: Decision trees and random forests

* Decision trees
* Random forests
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

# Define a new indicator variable for ‘excellent’ wines (quality score ≥ 8)
wines['excellent'] = wines.quality >= 8

# Prepare the data for use in scikit-learn
X = wines.drop(['quality', 'excellent'], axis=1)
y = wines.excellent.astype('int')

'''
Decision trees
'''

# Train a decision tree
tree1 = tree.DecisionTreeClassifier()
tree1.fit(X, y)

# Export the tree for plotting
tree.export_graphviz(tree1, 'tree1.dot', feature_names=X.columns)

# If you have Graphviz (http://www.graphviz.org) installed, run:
#     dot -Tpng tree1.dot -o tree1.png
# Alternatively, use WebGraphviz at http://www.webgraphviz.com/

# Define stratified folds for cross-validation
kf = cv.StratifiedKFold(y, n_folds=10, shuffle=True)

# Compute average AUC across folds
aucs = cv.cross_val_score(tree.DecisionTreeClassifier(),\
                          X, y, scoring='roc_auc', cv=kf)
np.mean(aucs)

# Train a decision tree by limiting:
# * the maximum number of questions (depth)
# * the minimum number of samples in each leaf
tree2 = tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf=50)
tree2.fit(X, y)

# Export the tree for plotting
tree.export_graphviz(tree2, 'tree2.dot', feature_names=X.columns)

# Investigate importances of predictors (the higher, the more important)
tree2.feature_importances_

'''
Random forests
'''

# Train a random forest with 20 decision trees
rf1 = ensemble.RandomForestClassifier(n_estimators=20)
rf1.fit(X, y)

# Investigate importances of predictors (the higher, the more important)
rf1.feature_importances_

# Evaluate performance through cross-validation
aucs = cv.cross_val_score(ensemble.RandomForestClassifier(n_estimators=20),\
                          X, y, scoring='roc_auc', cv=kf)
np.mean(aucs)

# What happens when we increase the number of trees?
for n_trees in [2, 5, 10, 20, 50, 100]:
    aucs = cv.cross_val_score(
        ensemble.RandomForestClassifier(n_estimators=n_trees), X, y,\
        scoring='roc_auc', cv=kf)
    print('{:>3} trees: mean AUC {:.2%}'.format(n_trees, np.mean(aucs)))

