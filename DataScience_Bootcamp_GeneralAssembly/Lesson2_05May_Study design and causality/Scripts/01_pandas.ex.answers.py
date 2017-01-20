#!/usr/bin/env python

'''
=== GA Data Science Q2 2016 ===

In-class exercise 1: Introduction to pandas
Model answers
'''

import numpy as np
import pandas as pd

AHP_URL = 'https://files.datapress.com/london/dataset/average-house-prices-borough/2016-05-04T15:05:51/average-house-prices-borough.xlsx'

INCOMES_URL = 'https://files.datapress.com/london/dataset/average-income-tax-payers-borough/2016-04-05T08:55:06/income-of-tax-payers.csv'

# Read in the ‘Median Annual’ sheet from the Average House Prices by Borough
# dataset at `AHP_URL` into a DataFrame called `ahp`
ahp = pd.read_excel(AHP_URL, sheetname='Median Annual')

# Print the ‘head’ and ‘tail’
ahp.head()
ahp.tail()

# Filter the DataFrame so that only Boroughs are included
ahp = ahp[ahp.Code.str.startswith('E09', na=False)]

# Set 'Code' as row names (index), then drop it from the DataFrame
ahp.index = ahp.Code
ahp.drop('Code', axis=1, inplace=True)

# Examine data types, and shape
ahp.dtypes
ahp.shape

# Calculate mean house prices by year
ahp.mean()

# BONUS: Calculate the first-order difference in mean house prices by year
np.diff(ahp.mean())

# We will now convert the dataset from ‘wide’ to ‘long’ format (‘melt’)
ahp = pd.melt(ahp, id_vars='Area', var_name='Year', value_name='Price')

# BONUS: How would you convert the dataset back to ‘long’ format?
ahp.pivot_table(values='Price', index='Area', columns='Year', aggfunc='sum')

# Convert 'Year' to integer
ahp['Year'] = ahp.Year.astype('int')

# Calculate mean house prices by year
ahp.groupby('Year').Price.mean()

# Calculate mean house prices by area using only data from 2010 onwards
ahp[ahp.Year >= 2010].groupby('Area').Price.mean()

# Identify the three areas with highest mean house prices
ahp.groupby('Area').Price.mean().sort_values(ascending=False).head(3)

# Read in the Average Income of Tax Payers by Borough from `INCOMES_URL` into a
# DataFrame called `incomes`
incomes = pd.read_csv(INCOMES_URL)

# Keep only the columns indicating the area and the medians for each year
incomes = incomes.iloc[:,(incomes.columns == 'Area') | incomes.columns.str.startswith('Median')]

# Rename the columns to only include the starting year (e.g. '1999-00' = 1999)
incomes.rename(columns=dict(zip(
    incomes.columns[1:],
    incomes.columns[1:].str.extract('(\\d+)-', expand=False)
)), inplace=True)

# ‘Melt’ the DataFrame
incomes = pd.melt(incomes, id_vars='Area', var_name='Year', value_name='Income')

# Convert 'Year' to integer
incomes['Year'] = incomes.Year.astype('int')

# Merge `incomes` with `ahp`, keeping only observations found in both
ahp = pd.merge(ahp, incomes, how='inner')

# Compute mean house prices and incomes by year
ahp.pivot_table(values=['Price', 'Income'], index='Year')

# BONUS: Compute the correlation between house prices and incomes
ahp.Price.corr(ahp.Income)
ahp.corr().ix['Price','Income']

# BONUS: Compute the correlation between house prices and incomes by area
ahp.groupby('Area').Price.corr(ahp.Income)

