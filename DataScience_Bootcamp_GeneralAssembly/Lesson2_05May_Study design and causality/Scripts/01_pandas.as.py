#!/usr/bin/env python

'''
=== GA Data Science Q2 2016 ===

Assignment 1: Introduction to pandas
'''

import os
import numpy as np
import pandas as pd

'''
Exercise 1
'''

# Read in the World Bank World Development Indicators data from `wbwdi.csv`
# into a DataFrame called `wbwdi`
wbwdi = pd.read_csv('wbwdi.csv', header = 0)

# Print the ‘head’ and ‘tail’
wbwdi.head()
wbwdi.tail()
# Examine the row names (index), data types, and shape
wbwdi.index
wbwdi.dtypes
wbwdi.shape
# Print the 'LIFEXP' Series
wbwdi['LIFEXP']
# Calculate the mean 'LIFEXP' for the entire dataset
wbwdi['LIFEXP'].mean()
# Count the number of occurrences of each 'Countrygp'
wbwdi['Countrygp'].value_counts()
# BONUS: Display only the number of rows of `wbwdi`
len(wbwdi)
# BONUS: Display the 3 most frequent values of 'Countrygp'
wbwdi['Countrygp'].value_counts().nlargest(3)
'''
Exercise 2
'''

# Filter `wbwdi` to only include African countries
wbwdi[wbwdi.Countrygp == 2.0]

# Filter `wbwdi` to only include African countries with LIFEXP > 60
wbwdi[(wbwdi.Countrygp == 2.0) & (wbwdi.LIFEXP > 60)]
# Calculate the mean 'LIFEXP' for all of Africa
africa = wbwdi[wbwdi.Countrygp == 2.0]
africa['LIFEXP'].mean()
# Determine which 10 countries have the highest LIFEXP
wbwdi.sort('LIFEXP', ascending = False).head(10)
# BONUS: Sort `wbwdi` by 'Countrygp' and then by 'LIFEXP' (in a single command)
wbwdi.sort(['Countrygp', 'LIFEXP'], ascending = [False, False])
# BONUS: Filter `wbwdi` to only include African or Middle Eastern countries
#        without using `|`.


'''
Exercise 3
'''

# Count the number of missing values in each column of `wbwdi`
wbwdi.isnull().sum() 
# Show only countries for which 'LIFEXP' is missing
wbwdi[wbwdi.LIFEXP.isnull()]
# How many rows remain if you drop all rows with any missing values?
len(wbwdi.dropna())
# BONUS: Create a new column called 'initial' that contains the first letter of
#        the country name (e.g., 'A' for Afghanistan)
wbwdi['initial'] = wbwdi['Country'].str[:1]

'''
Exercise 4
'''

# Calculate the mean 'LIFEXP' by 'Countrygp'
wbwdi.groupby('Countrygp').LIFEXP.mean()
# Calculate the minimum and maximum 'LIFEXP' by 'Countrygp'
wbwdi.groupby('Countrygp').LIFEXP.agg(['min', 'max'])
# BONUS: Cross-tabulate 'Countrygp' and 'initial'
pd.crosstab(wbwdi.Countrygp, wbwdi.initial)

# BONUS: Calculate the median 'LIFEXP' for each combination of 'Countrygp' and
#        'initial'
wbwdi.pivot_table(values='LIFEXP', index='Countrygp', columns='initial', aggfunc='median')
