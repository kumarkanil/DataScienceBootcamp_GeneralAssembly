#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 16: Bayesian change detection
'''

import numpy as np
import pymc as pm

import matplotlib.pyplot as plt

%matplotlib qt

# Generate some data from a Poisson distribution with a change point
y = np.concatenate((
    np.random.poisson(5, size=100),
    np.random.poisson(10, size=200)
))

# Plot generated data
plt.plot(y)

# Define prior distributions
change_point = pm.DiscreteUniform('change_point', lower=1, upper=len(y) - 1)
early_rate = pm.Exponential('early_rate', beta=1.0)
late_rate = pm.Exponential('late_rate', beta=1.0)

# Define (observed) stochastic variable for the number of arrivals
@pm.stochastic(observed=True, dtype=int)
def arrivals(value=y, change_point=change_point,\
             early_rate=early_rate, late_rate=late_rate):
    return pm.poisson_like(value[:change_point], early_rate) +\
           pm.poisson_like(value[change_point:], late_rate)

# Create model and sample
model = pm.MCMC([change_point, early_rate, late_rate, arrivals])
model.sample(iter=100000, burn=10000, thin=100)

# Explore posterior summary statistics
model.stats()
model.summary()

# Plot traces and posterior densities
pm.Matplot.plot(model)

