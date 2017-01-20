#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 14: Dimensionality reduction

* Multidimensional scaling (MDS)
* Principal component analysis (PCA)
* Partial least squares (PLS) regression
'''

import numpy as np
import pandas as pd

from sklearn import preprocessing, cross_validation as cv, grid_search,\
                    manifold, decomposition as dec,\
                    cross_decomposition as cross_dec

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

%matplotlib qt

BOROUGHS_URL = 'https://files.datapress.com/london/dataset/london-borough-profiles/2015-09-24T15:49:52/london-borough-profiles.csv'

# Read in the London Borough Profiles datasets
boroughs = pd.read_csv(BOROUGHS_URL)

# Filter the DataFrame so that only boroughs are included
boroughs = boroughs[boroughs.Code.str.startswith('E09', na=False)]

# Select columns of interest
boroughs = boroughs[[
    'Area name',
    'Population density (per hectare) 2015',
    'Proportion of population aged 0-15, 2015',
    'Proportion of population of working-age, 2015',
    'Proportion of population aged 65 and over, 2015',
    '% of resident population born abroad (2014)',
    'Unemployment rate (2014)',
    'Gross Annual Pay, (2014)',
    'Modelled Household median income estimates 2012/13',
    'Number of active businesses, 2013',
    'Two-year business survival rates (started in 2011)',
    'Crime rates per thousand population 2014/15',
    'Fires per thousand population (2014)',
    'Ambulance incidents per hundred population (2014)',
    'Median House Price, 2014',
    '% of area that is Greenspace, 2005',
    'Total carbon emissions (2013)',
    'Household Waste Recycling Rate, 2013/14',
    'Number of cars, (2011 Census)',
    'Number of cars per household, (2011 Census)',
    '% of adults who cycle at least once per month, 2013/14',
    'Average Public Transport Accessibility score, 2014',
    'Male life expectancy, (2011-13)',
    'Female life expectancy, (2011-13)',
    'Teenage conception rate (2013)',
    'Life satisfaction score 2011-14 (out of 10)',
    'Worthwhileness score 2011-14 (out of 10)',
    'Happiness score 2011-14 (out of 10)',
    'Anxiety score 2011-14 (out of 10)',
    'Childhood Obesity Prevalance (%) 2013/14',
    'People aged 17+ with diabetes (%)',
    'Mortality rate from causes considered preventable'
]]

# Set row names (index)
boroughs.set_index('Area name', inplace=True)

# Remove boroughs with missing values
boroughs.dropna(inplace=True)

# Extract information on ‘feelings’
col_idx = [
    'Life satisfaction score 2011-14 (out of 10)',
    'Worthwhileness score 2011-14 (out of 10)',
    'Happiness score 2011-14 (out of 10)',
    'Anxiety score 2011-14 (out of 10)'
]
feelings = boroughs[col_idx]
boroughs.drop(col_idx, axis=1, inplace=True)

'''
Multidimensional scaling (MDS)
'''

# Create a pipeline that scales the data and performs MDS
smds = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('mds', manifold.MDS())
])

# Two-dimensional projection (‘embedding’) of 'boroughs'
boroughs_mds = smds.fit_transform(boroughs)

fig, ax = plt.subplots()
ax.scatter(boroughs_mds[:,0], boroughs_mds[:,1])
for i, name in enumerate(boroughs.index):
    ax.annotate(name, boroughs_mds[i,:])

'''
Principal component analysis (PCA)
'''

# Create a pipeline that scales the data and performs PCA
spca = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('pca', dec.PCA())
])

# Scores (projection of 'boroughs' on the PCs)
scores = spca.fit_transform(boroughs)

# Scores plot
fig, ax = plt.subplots()
ax.scatter(scores[:,0], scores[:,1])
for i, name in enumerate(boroughs.index):
    ax.annotate(name, scores[i,0:2])

# Loadings (coefficients defining the PCs)
spca.named_steps['pca'].components_

# Explained variance
spca.named_steps['pca'].explained_variance_
np.cumsum(spca.named_steps['pca'].explained_variance_)

# Explained variance ratio
spca.named_steps['pca'].explained_variance_ratio_

# Scree plot
plt.bar(np.arange(1, spca.named_steps['pca'].n_components_ + 1) - 0.4,\
        spca.named_steps['pca'].explained_variance_ratio_)
cum_evr = np.cumsum(spca.named_steps['pca'].explained_variance_ratio_)
plt.plot(np.arange(1, spca.named_steps['pca'].n_components_ + 1), cum_evr,\
         color='black')

'''
Partial least squares (PLS) regression
'''

# Create a pipeline that scales the data and performs PLS regression
spls = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('pls', cross_dec.PLSRegression(scale=False))
])

# Train a PLS regression model with three components
spls.set_params(
    pls__n_components=3
)
spls.fit(boroughs, feelings)

# Define folds for cross-validation
kf = cv.KFold(len(feelings), n_folds=10, shuffle=True)

# Compute average MSE across folds
mses = cv.cross_val_score(spls, boroughs, feelings,\
                          scoring='mean_squared_error', cv=kf)
np.mean(-mses)

# Determine ‘optimal’ number of components
gs = grid_search.GridSearchCV(
    estimator=spls,
    param_grid={
        'pls__n_components': np.arange(1, 10)
    },
    scoring='mean_squared_error',
    cv=kf
)
gs.fit(boroughs, feelings)

-gs.best_score_
gs.best_estimator_
gs.grid_scores_

# Plot number of components against MSE
plt.plot(np.arange(1, 10), [ -x[1] for x in gs.grid_scores_ ])

