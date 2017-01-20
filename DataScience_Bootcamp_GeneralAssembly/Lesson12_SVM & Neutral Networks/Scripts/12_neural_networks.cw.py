#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 12: Neural networks
'''

# Install latest versions of Theano and Keras manually:
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# pip install --upgrade --no-deps git+git://github.com/fchollet/keras.git

try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin

import numpy as np
import pandas as pd

from sklearn import preprocessing, metrics

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

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
X = wines.drop(['quality', 'red'], axis=1).get_values()
y = wines.red.astype('int').get_values()

# Scale X
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Initialise neural network
nn = Sequential()

# Input layer feeding into hidden layer with 5 neurons (sigmoid activation)
nn.add(Dense(input_dim=X.shape[1], output_dim=5, activation='sigmoid'))

# Hidden layer feeding into a single output neuron (sigmoid activation)
nn.add(Dense(output_dim=1, activation='sigmoid'))

# Use logistic loss
nn.compile(loss='binary_crossentropy', optimizer='adam')

# Inspect weights before training
nn.get_weights()

# Train the networks
nn.fit(X_scaled, y, batch_size=1, nb_epoch=10, validation_split=0.2)

# Inspect weights after training
nn.get_weights()

# Use network to predict probabilities
pred_probs = nn.predict(X_scaled)[:,0]

# Compute AUC
metrics.roc_auc_score(y, pred_probs)

