#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 7: Classification

* Visualisation
* Confusion matrix
* k-nearest neighbours classifier
'''

import numpy as np
import pandas as pd

from sklearn import metrics, neighbors, cross_validation as cv, grid_search

import seaborn as sns

%matplotlib qt

IRIS_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data'

var_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Read in Iris dataset
iris = pd.read_csv(IRIS_URL, header=None, names=var_names)

# Basic data exploration
iris.head()
iris.describe()
iris.isnull().sum()
iris.species.value_counts()

'''
Visualisation
'''

# Look for differences between species
iris.groupby('species').mean()
iris.groupby('species').describe()

# Box plot of all numeric columns grouped by species
iris.boxplot(by='species')

# Scatter matrix of all predictors coloured by species
sns.pairplot(iris, hue='species')

# Box plot of petal length by species
sns.boxplot(x='species', y='petal_length', data=iris)

# Alternatively…
sns.stripplot(x='species', y='petal_length', data=iris, jitter=True)
sns.swarmplot(x='species', y='petal_length', data=iris)
sns.violinplot(x='species', y='petal_length', data=iris)

# Scatter plot of petal length versus petal width with density estimates
sns.jointplot(x='petal_length', y='petal_width', data=iris, kind='kde')

# Define new predictor (petal area)
iris['petal_area'] = iris.petal_length * iris.petal_width

# Look for differences in petal area between species
iris.groupby('species').petal_area.describe().unstack()

# Box plot of petal area by species
sns.boxplot(x='species', y='petal_area', data=iris)

# Extract observations that cannot be linearly separated
iris[(iris.petal_area >= 7.5) & (iris.petal_area <= 8.64)].sort_values('petal_area')

'''
Confusion matrix
'''

# Define a function to predict the species based on petal area only
def classify_iris(row):
    if row.petal_area < 2:
        return 'Iris-setosa'
    elif row.petal_area < 7.5:
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'

# Add predictions to DataFrame
iris['prediction'] = iris.apply(classify_iris, axis=1)

# Compute confusion matrix
metrics.classification.confusion_matrix(iris.species, iris.prediction)
pd.crosstab(iris.species, iris.prediction)

# Print classification report
print(metrics.classification_report(iris.species, iris.prediction))

'''
k-nearest neighbours classifier
'''

# Prepare data
X = iris.loc[:,'sepal_length':'petal_width']
y = iris.species.factorize()[0]

# All points in the neighbourhood are weighted equally
knn_uniform = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_uniform.fit(X, y)

# Compute accuracy
metrics.accuracy_score(y, knn_uniform.predict(X))
np.mean(y == knn_uniform.predict(X))

# Points in the neighbourhood are weighted by the inverse of the distance
knn_distance = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_distance.fit(X, y)

metrics.accuracy_score(y, knn_distance.predict(X))

# Determine ‘optimal’ number of neighbours and method
params = {
    'weights': ['uniform', 'distance'],
    'n_neighbors': np.arange(2, 50)
}
kf = cv.StratifiedKFold(iris.species, n_folds=5, shuffle=True)
gs = grid_search.GridSearchCV(
    estimator=neighbors.KNeighborsClassifier(),
    param_grid=params,
    cv=kf
)
gs.fit(X, y)

# ‘Best’ accuracy
gs.best_score_

# ‘Best’ model
gs.best_estimator_

# Confusion matrix and classification report of ‘best’ model
metrics.classification.confusion_matrix(y, gs.best_estimator_.predict(X))
print(metrics.classification_report(y, gs.best_estimator_.predict(X)))

# All grid configurations and corresponding performances
gs.grid_scores_

