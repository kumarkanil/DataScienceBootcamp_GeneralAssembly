#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 2: Visualisation with matplotlib and Seaborn
Model answers
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib qt

FHRS_URL = 'https://opendata.camden.gov.uk/api/views/ggah-dkrr/rows.csv?accessType=DOWNLOAD'

# Read in the Food Hygiene Rating Scheme data from `FHRS_URL` into a DataFrame
# called `fhrs`
fhrs = pd.read_csv(FHRS_URL)

# Change the data type of 'Rating Date' to `datetime`
fhrs['Rating Date'] = pd.to_datetime(fhrs['Rating Date'])

# Filter `fhrs` to include only restaurants/cafés/canteens that are not exempt,
# and are not awaiting a new rating
fhrs = fhrs[(fhrs['Business Type Description'] == 'Restaurant/Cafe/Canteen') &\
            (fhrs['Rating Value'] != 'Exempt') &\
            (~fhrs['New Rating Pending'])]

# Change the data type of 'Rating Value' to 'int'
fhrs['Rating Value'] = fhrs['Rating Value'].astype('int')

# Produce a bar plot of 'Rating Value'
fhrs['Rating Value'].value_counts().sort_index().plot.bar()
sns.countplot(x='Rating Value', data=fhrs)

# Create a new variable 'Rating Year' from 'Rating Date'
fhrs['Rating Year'] = fhrs['Rating Date'].map(lambda x: x.year)

# Produce a box plot of 'Rating Value' grouped by 'Rating Year'
fhrs.boxplot(column='Rating Value', by='Rating Year')
sns.boxplot(x='Rating Year', y='Rating Value', data=fhrs)

# Produce a scatter plot of 'Hygiene Score', 'Structural Score', 'Confidence In Management Score', and 'Rating Value'
scores = ['Hygiene Score', 'Structural Score', 'Confidence In Management Score', 'Rating Value']
pd.scatter_matrix(fhrs[scores])
sns.pairplot(fhrs[scores].dropna())

# BONUS: Using Seaborn, produce a scatter plot of 'Hygiene Score' against
#        'Rating Value' including a linear regression line.
#        Add some jitter to prevent overplotting.
sns.lmplot(x='Hygiene Score', y='Rating Value', data=fhrs, x_jitter=1.25, y_jitter=0.25)

