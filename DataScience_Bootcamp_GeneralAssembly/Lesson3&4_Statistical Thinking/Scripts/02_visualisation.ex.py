#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 2: Visualisation with matplotlib and Seaborn
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

# Change the data type of 'Rating Date' to `datetime`

# Filter `fhrs` to include only restaurants/cafés/canteens that are not exempt,
# and are not awaiting a new rating

# Change the data type of 'Rating Value' to 'int'

# Produce a bar plot of 'Rating Value'

# Create a new variable 'Rating Year' from 'Rating Date'

# Produce a box plot of 'Rating Value' grouped by 'Rating Year'

# Produce a scatter plot of 'Hygiene Score', 'Structural Score', 'Confidence In Management Score', and 'Rating Value'

# BONUS: Using Seaborn, produce a scatter plot of 'Hygiene Score' against
#        'Rating Value' including a linear regression line.
#        Add some jitter to prevent overplotting.

