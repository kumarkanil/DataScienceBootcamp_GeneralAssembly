#!/usr/bin/env python

'''
=== GA Data Science Q2 2016 ===

Code walk-through 1: Introduction to pandas

* Reading files, selecting columns, and summarising
* Filtering and sorting
* Selecting, renaming, adding, and removing columns
* Handling missing values (NAs)
* Split-apply-combine
* Pivot tables
* Merging DataFrames
* Other useful features
* Less frequently used features
'''

import os

import numpy as np
import pandas as pd

'''
Reading files, selecting columns, and summarising
'''

# Read in the British Social Attitudes 2014 data
# Note: pandas can read data from a local file or directly from a URL
bsa2014 = pd.read_csv(os.path.join('datasets', 'bsa2014.csv'), index_col=0)

# Examine the BSA2014 data
bsa2014                                # Prints the first 30 and last 30 rows
type(bsa2014)                          # DataFrame
bsa2014.head()                         # Prints the first 5 rows (by default)
bsa2014.ix[0:4]                        # (as above)
bsa2014.iloc[0:4]                      # (as above)
bsa2014.head(10)                       # Prints the first 10 rows
bsa2014.tail()                         # Prints the last 5 rows (by default)
bsa2014.index                          # Row labels (an index)
bsa2014.columns                        # Column names (another index)
bsa2014.dtypes                         # Data type of each column
len(bsa2014)                           # Number of rows
bsa2014.shape                          # Shape (number of rows and columns)
bsa2014.values                         # Underlying NumPy array
type(bsa2014.values)                   # ndarray

# Select a column
bsa2014['RSex']                        # Select one column
bsa2014.RSex                           # (as above)
type(bsa2014['RSex'])                  # Series

# Summarise (describe) the DataFrame
bsa2014.describe()                     # Describes all numeric-typed columns
bsa2014.describe(include=['float'])    # Describes only continuous variables

# Summarise a Series
bsa2014.RAge.describe()                # Describes a single column
bsa2014.RAge.mean()                    # Calculates the mean

# Count the number of occurrences of each value (tabulate)
bsa2014.RSex.value_counts()            # Most useful for categorical variables
bsa2014.RAge.value_counts()            # Can be used with continuous variables

'''
Filtering and sorting
'''

# Boolean filtering: only show participants aged 20 years or younger
young_idx = bsa2014.RAge < 20          # Creates a Series of Booleans...
young_idx.dtype
bsa2014[young_idx]                     # ...that can be used to filter rows
bsa2014[bsa2014.RAge < 20]             # (also in a single step)
bsa2014[bsa2014.RAge < 20].Country     # Selects one column from the filtered DF
bsa2014[bsa2014.RAge < 20].Country.value_counts()

# Boolean filtering: multiple conditions
bsa2014[(bsa2014.RAge < 20) & (bsa2014.Country == 2)]
bsa2014[(bsa2014.RAge < 20) | (bsa2014.RAge > 60)]

# Alternatively...
bsa2014.query('RAge < 20 and Country == 2')
bsa2014.query('RAge < 20 or RAge > 60')

# `loc` selects rows (and columns, see below) by name (index)
bsa2014.loc[160002, :]                 # Participant 160002, all columns

# `iloc` selects rows (and columns, see below) by integer position
bsa2014.iloc[1, :]                     # First row, all columns
bsa2014.iloc[0:3, :]                   # First three rows, all columns

# Sorting values
bsa2014.sort_values('RAge')            # Sorts a DF by a single column
bsa2014.sort_values('RAge', ascending=False)
bsa2014.sort_values('RAge', inplace=True)

# Sorting by index
bsa2014.RAge.value_counts().sort_index()

'''
Selecting, renaming, adding, and removing columns
'''

# Select multiple columns
my_cols = ['RSex', 'RAge']             # Creates a list of column names...
bsa2014[my_cols]                       # ...that can be used to select columns
bsa2014[['RSex', 'RAge']]              # (also in a single step)

# Select multiple columns by name using `loc`
bsa2014.loc[:, 'RSex']                 # All rows, one column
bsa2014.loc[:, ['RSex', 'RAge']]       # All rows, two columns
bsa2014.loc[:, 'Country':'RAge']       # All rows, range of columns

# Select multiple columns by integer position using `iloc`
bsa2014.iloc[:, 5]                     # All rows, one column
bsa2014.iloc[:, [5, 6]]                # All rows, two columns
bsa2014.iloc[:, 4:6]                   # All rows, range of columns

# Rename one or more columns
bsa2014.rename(columns={
    'RAge': 'age',
    'RSex': 'sex'
}, inplace=True)

# Add a new column as a function of existing columns
bsa2014['under20'] = bsa2014.age < 20

# Alternatively...
bsa2014.assign(under20 = lambda x: x.age < 20)

# Removing (dropping) columns (axis=1)
bsa2014.drop('under20', axis=1, inplace=True)

'''
Handling missing values (NAs)
'''

# Missing values may be coded using ‘special’ values: check the data dictionary
bsa2014.age.replace(99, np.NaN, inplace=True)

bsa2014.sort_values('age')             # Sorts a DF by a single column

# Find missing values in a Series
bsa2014.age.isnull()                   # True if missing
bsa2014.age.notnull()                  # True if not missing

bsa2014[bsa2014.age.isnull()]          # Only show rows where age is missing
bsa2014[bsa2014.age.notnull()]         # Only show rows where age is not missing

# Side note: adding Booleans
bsa2014.age.isnull().sum()             # Number of missing values

# Find missing values in a DataFrame
bsa2014.isnull()                       # Returns a DataFrame of Booleans
bsa2014.isnull().sum()                 # Counts missing values in each column

# Drop missing values
bsa2014.dropna()                       # Drops a row if any values are missing
bsa2014.dropna(how='all')              # Drops a row if all values are missing

# Fill in missing values
bsa2014.age.fillna(value='NA')

# See also parameters `na_values` and `na_filter` to `read_csv`

'''
Split-apply-combine
'''

# For each country, count the number of participants
bsa2014.Country.value_counts()

# For each country, compute the mean age
bsa2014.groupby('Country').age.mean()

# For each country, describe age
bsa2014.groupby('Country').age.describe()

# Similar, but outputs a customisable DataFrame
bsa2014.groupby('Country').age.agg(['count', 'median', 'min', 'max'])
bsa2014.groupby('Country').age.agg(['count', 'median', 'min', 'max']).sort_values('max')

'''
Pivot tables
'''

# Average age by country and sex (in rows)
bsa2014.pivot_table(values='age', index=['Country', 'sex'])

# Median age by country and sex (in rows and columns, respectively)
bsa2014.pivot_table(values='age', index='Country', columns='sex', aggfunc='median')

# Average age by country, income group, and sex
bsa2014.pivot_table(values='age', index=['Country', 'incomegp'], columns='sex')

'''
Merging DataFrames
'''

A = pd.DataFrame({
    'color': ['Blue', 'Green', 'Red'],
    'num': [1, 2, 3]
})

B = pd.DataFrame({
    'color': ['Blue', 'Green', 'Yellow'],
    'size': ['S', 'M', 'L']
})

# Inner join: include only observations found in both A and B
pd.merge(A, B, how='inner')

# Outer join: include observations found in either A or B
pd.merge(A, B, how='outer')

# Left join: include all observations found in A
pd.merge(A, B, how='left')

# Right join: include all observations found in B
pd.merge(A, B, how='right')

'''
Other useful features
'''

# Map existing values to a different set of values
bsa2014['country_name'] = bsa2014.Country.map({
    1: 'england',
    2: 'scotland',
    3: 'wales'
})

# Encode strings as integers (starting at 0, NA = -1 by default)
bsa2014['country_idx'] = bsa2014.country_name.factorize()[0]

# Determine unique values in a Series
bsa2014.country_name.nunique()         # Counts the number of unique values
bsa2014.country_name.unique()          # Returns the unique values

# Replace all instances of a value in a Series
bsa2014.country_name.replace({
    'england': 'England',
    'scotland': 'Scotland',
    'wales': 'Wales'
}, inplace=True)

# String methods can be accessed via `str`
bsa2014.country_name.str.upper()
bsa2014.country_name.str.contains('Scot').sum()

# To convert a string to datetime, use `pd.to_datetime`

# Set and remove row names (index)
bsa2014.set_index('sex', inplace=True)
bsa2014.reset_index(inplace=True)

# Change the data type of a column
# (see also parameter `dtype` to `read_csv` to do so when reading in a file)
bsa2014['age'] = bsa2014.age.astype('float')

# Create dummy variables for country, excluding first column
country_dummies = pd.get_dummies(bsa2014.Country, prefix='country', drop_first=True)

# Concatenate two DataFrames (axis=0 for rows, axis=1 for columns)
bsa2014 = pd.concat([bsa2014, country_dummies], axis=1)

'''
Less frequently used features
'''

# Create a DataFrame from a dictionary
uk_countries = pd.DataFrame({
    'country': ['England', 'Scotland', 'Wales', 'Northern Ireland'],
    'capital': ['London', 'Edinburgh', 'Cardiff', 'Belfast']
})

# Create a DataFrame from a list of lists
uk_countries = pd.DataFrame([
    ['England', 'London'],
    ['Scotland', 'Edinburgh'],
    ['Wales', 'Cardiff'],
    ['Northern Ireland', 'Belfast']
], columns=['country', 'capital'])

# Detect duplicate rows
bsa2014.duplicated()                   # True if a row has been seen previously
bsa2014.duplicated().sum()             # Count of duplicates
bsa2014.drop_duplicates()              # Drops duplicate rows
bsa2014.age.duplicated()               # Checks a single column for duplicates
bsa2014.duplicated(['Country','sex','age']).sum() # ...or multiple

# Cross-tabulate two Series
pd.crosstab(bsa2014.Country, bsa2014.sex)

# Display the memory usage of a DataFrame
bsa2014.info()                         # Total
bsa2014.memory_usage()                 # By column

# Change a Series to the `category` data type
# (lower memory usage, higher performance)
bsa2014['Country'] = bsa2014.Country.astype('category')

# Write a DataFrame out in CSV format
bsa2014.to_csv('bsa2014_updated.csv')

# Randomly sample a DataFrame (e.g. 75%/25% split)
train = bsa2014.sample(frac=0.75)
test = bsa2014[~bsa2014.index.isin(train.index)]

# Change the maximum number of rows and columns printed (None = unlimited)
pd.set_option('max_rows', 10)          # Default is 60 rows
pd.set_option('max_columns', None)     # Default is 20 columns
bsa2014

# Reset options to defaults
pd.reset_option('max_rows')
pd.reset_option('max_columns')

# Change the options temporarily (within the `with` block)
with pd.option_context('max_rows', 10, 'max_columns', None):
    print(bsa2014)

