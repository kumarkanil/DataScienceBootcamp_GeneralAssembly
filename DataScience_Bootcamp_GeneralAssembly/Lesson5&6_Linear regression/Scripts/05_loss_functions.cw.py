#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 5: Loss functions

* Guessing values
* MAE and (R)MSE
'''

import numpy as np
import seaborn as sns

%matplotlib qt

'''
Guessing values
'''

# Simulate 1,000 observations to use as training set (this is our random sample)
training = np.random.gamma(3, 1, 1000)

# Plot the distribution of the sample
sns.distplot(training)

# Suppose we want to predict future values using a single number.
# What value should we use?

# Simulate 100 ‘future’ values from the same population to use as test set
test = np.random.gamma(5, 1, 100)

# Before we can answer the question above, we need to define a ‘loss’ function;
# we will start by investigating the squared loss.

def squared_loss(x, g):
    return np.sum(np.square(x - g))

# We now define a list of 1,000 ‘guesses’ for our prediction between 0 and 10
guesses = np.arange(0, 10, 0.01)

# What are the values of the loss function for each guess?
y_squared = [ squared_loss(test, g) for g in guesses ]

# Plot guesses versus loss
sns.pointplot(guesses, y_squared, markers='')

# Where is the minimum of the loss function?
guesses[np.argmin(y_squared)]
np.mean(test)

# In general, our ‘guess’ would have to be estimated from the training data;
# this is fine as long as we can assume that the past will repeat itself.
np.mean(training)

# Let’s now consider the absolute loss function and repeat!

def absolute_loss(x, g):
    return np.sum(np.abs(x - g))

# What are the values of the loss function for each guess?
y_absolute = [ absolute_loss(test, g) for g in guesses ]

# Plot guesses versus loss
sns.pointplot(guesses, y_absolute, markers='')

# Where is the minimum of the loss function?
guesses[np.argmin(y_absolute)]
np.median(test)

# How does it differ from the estimate obtained using the training set?
np.median(training)

'''
MAE and (R)MSE
'''

# Example of true and predicted responses
true = np.array([4, 5, 5, 7, 10])
pred = np.array([8, 10, 5, 6, 8])

# Compute MAE
np.mean(np.abs(true - pred))
metrics.mean_absolute_error(true, pred)

# Compute MSE
np.mean(np.square(true - pred))
metrics.mean_squared_error(true, pred)

# Compute RMSE
np.sqrt(np.mean(np.square(true - pred)))
np.sqrt(metrics.mean_squared_error(true, pred))

# (R)MSE penalises larger errors more than MAE
true = np.array([4, 5, 5, 7, 10])
pred = np.array([6, 6, 5, 6, 2])
metrics.mean_absolute_error(true, pred)
metrics.mean_squared_error(true, pred)

