#!/usr/bin/env python

'''
=== GA Data Science Q2 2016 ===

Code walk-through 2: Summary statistics
'''

import numpy as np
import pandas as pd

df = pd.DataFrame({
    'example1': [18, 24, 17, 21, 24, 16, 29, 18],
    'example2': [75, 87, 49, 68, 75, 84, 98, 92],
    'example3': [55, 47, 38, 66, 56, 64, 44, 39]
})

# Number of values
df.count()

# Mean
df.sum() / df.count()                  # Manually
df.mean()                              # Built-in function

# Mode(s)
df.example1.value_counts()             # Tabulate values
df.mode()                              # Find mode(s)

# Order statistics
df.min()                               # Minimum (0% quantile)
df.max()                               # Maximum (100% quantile)
df.median()                            # Median (50% quantile)
df.quantile(np.linspace(0, 1, 5))      # 0%, 25%, 50%, 75%, 100% quantiles

# All-in-one solution!
df.describe()

# Variance: E[X^2] - E[X]^2
(np.square(df - df.mean())).sum() / (df.count() - 1)
df.var()

# Standard deviation
np.sqrt((np.square(df - df.mean())).sum() / (df.count() - 1))
df.std()

# Covariance between two variables: E[XY] - E[X] * E[Y]
# By subtracting the mean, we can avoid computing the second part
df0 = df - df.mean()
(df0.example1 * df0.example2).sum() / (df0.example1.count() - 1)

df.cov()

# Correlation between two variables: Cov[X,Y] / sqrt(V[X] * V[Y])
numerator = (df0.example1 * df0.example2).sum() / (df0.example1.count() - 1)
denominator = df.example1.std() * df.example2.std()
numerator / denominator

df.corr()

