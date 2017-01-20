#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 3: Hypothesis testing
'''

import numpy as np
import statsmodels.stats.proportion as smp
import seaborn as sns

from scipy.stats import binom, binom_test

%matplotlib qt

# Record here the number of orange and total M&M’S® in your sample
n_orange = 11
n = 46

# Proportion of orange M&M’S®
p_orange = n_orange / n

# 95% confidence interval for the proportion
smp.proportion_confint(n_orange, n, alpha=0.05)

# What happens if we require a 99% (instead of 95%) CI?
smp.proportion_confint(n_orange, n, alpha=0.01)

# What would happen if we had 10 times as many M&M’S® (but the same proportion)?
smp.proportion_confint(n_orange * 10, n * 10, alpha=0.05)

# Assuming that the population-level proportion is 20%, what is the probability
# of observing our proportion?
# (`pmf` returns the ‘probability mass function’, i.e. the probability at each
#  discrete value 0, ..., n)
binom.pmf(n_orange, n, 0.2)

# Visualise the probability at each discrete value 0, ..., n
x = np.arange(n + 1)
y = binom.pmf(x, n, 0.2)
sns.pointplot(x, y)

# What is the sum of all these probabilities?
binom.pmf(x, n, 0.2).sum()

# Test whether the proportion is ‘significantly different’ from 20%
# (`binom_test` returns the p-value for the test)
binom_test(n_orange, n, 0.2)

# What happens if we change the ‘null’ proportion we are testing against?
p = np.linspace(0, 1, 51)
y = [ binom_test(n_orange, n, p) for p in p ]
sns.pointplot(p, y)

# When p-values are very small, it is customary to apply a -log10 transformation
sns.pointplot(p, -np.log10(y))

