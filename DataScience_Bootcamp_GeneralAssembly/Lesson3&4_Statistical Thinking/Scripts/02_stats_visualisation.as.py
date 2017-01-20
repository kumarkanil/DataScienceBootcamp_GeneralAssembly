#!/usr/bin/env python

'''
GA Data Science Q2 2016

Assignment 2: Summary statistics and visualisation
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

%matplotlib qt

'''
Exercise 1
'''

# Read in the General Household Survey 2005 data

# Change the data type of 'sex' to `int`

# Recode missing values in the variables 'drating' and 'workhrs'

# Compute the mode of 'drating'

# Compute the mean and standard deviation of 'drating'

# Compute the median of 'drating'

# Repeat excluding zeros

'''
Exercise 2
'''

# Visualise the distribution of 'drating' (histogram and density estimate)

# Repeat excluding zeros

# Repeat excluding zeros and applying a logarithmic transformation

# Produce a box plot of 'drating' grouped by 'sex'

# BONUS: Repeat after applying a logarithmic transformation to 'drating'

# Produce a scatter plot of 'drating', 'age', and 'workhrs'

# Compute the correlation matrix of 'drating', 'age', and 'workhrs'

# BONUS: Represent the correlation matrix using a heat map

# BONUS: Formally test the hypothesis that 'drating' differs by 'sex' using a
#        two-sample t-test.

# BONUS: Repeat after applying a logarithmic transformation.

# BONUS: Repeat using a Wilcoxon rank-sum test, and observe the effect of the
#        logarithmic transformation in this case.

