#!/usr/bin/env python

'''
=== GA Data Science Q2 2016 ===

In-class exercise 1: Introduction to pandas
'''
import os
import numpy as np
import pandas as pd

AHP_URL = 'https://files.datapress.com/london/dataset/average-house-prices-borough/2016-05-04T15:05:51/average-house-prices-borough.xlsx'

INCOMES_URL = 'https://files.datapress.com/london/dataset/average-income-tax-payers-borough/2016-04-05T08:55:06/income-of-tax-payers.csv'

# Read in the ‘Median Annual’ sheet from the Average House Prices by Borough
# dataset at `AHP_URL` into a DataFrame called `ahp`

# Print the ‘head’ and ‘tail’

# Filter the DataFrame so that only Boroughs are included

# Set 'Code' as row names (index), then drop it from the DataFrame

# Examine data types, and shape

# Calculate mean house prices by year

# BONUS: Calculate the first-order difference in mean house prices by year

# We will now convert the dataset from ‘wide’ to ‘long’ format (‘melt’)
ahp = pd.melt(ahp, id_vars='Area', var_name='Year', value_name='Price')

# BONUS: How would you convert the dataset back to ‘long’ format?

# Convert 'Year' to integer

# Calculate mean house prices by year

# Calculate mean house prices by area using only data from 2010 onwards

# Identify the three areas with highest mean house prices

# Read in the Average Income of Tax Payers by Borough from `INCOMES_URL` into a
# DataFrame called `incomes`

# Keep only the columns indicating the area and the medians for each year

# Rename the columns to only include the starting year (e.g. '1999-00' = 1999)

# ‘Melt’ the DataFrame

# Convert 'Year' to integer

# Merge `incomes` with `ahp`, keeping only observations found in both

# Compute mean house prices and incomes by year

# BONUS: Compute the correlation between house prices and incomes

# BONUS: Compute the correlation between house prices and incomes by area

