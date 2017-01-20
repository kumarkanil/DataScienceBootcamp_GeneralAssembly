#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 3: Simulation of confidence intervals
'''

import numpy as np
import statsmodels.stats.api as sms

n = 1000000                            # Population size
mu = 5                                 # Population-level mean
sigma = 10                             # Population-level SD

alpha = 0.05                           # Significance level (1 - confidence level)
sample_size = 100                      # Sample size
n_simulations = 100                    # Number of simulations

# Simulate the population
population = np.random.normal(mu, sigma, n)

# Check population-level mean and SD
population.mean()
population.std()

# Initialise counter for the number of simulations where the CI contains the
# population-level mean (expected to be (1-alpha)% of n_simulations)
contained = 0

for i in range(n_simulations):
    # Sample from the population without replacement
    sample = np.random.choice(population, sample_size, replace=False)

    # Compute confidence interval for the mean
    conf_int = sms.DescrStatsW(sample).zconfint_mean(alpha)

    if conf_int[0] < mu < conf_int[1]:
        contained += 1

print(contained / n_simulations)

