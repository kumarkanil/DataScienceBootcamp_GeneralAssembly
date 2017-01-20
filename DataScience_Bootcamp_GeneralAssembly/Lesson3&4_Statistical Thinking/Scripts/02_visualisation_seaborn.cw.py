#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 2: Visualisation with Seaborn

* Histograms and density plots
* Scatter plots
* Bar plots
* Box plots
* Line plots
* Saving plots to disk
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Check the documentation for '%matplotlib' to be able to work interactively
%matplotlib qt

# Increase figure and font sizes for easier viewing
sns.set(rc={
    'figure.figsize': (8, 6),
    'font.size': 14
})

# Read in the World Bank World Development Indicators data
wbwdi = pd.read_csv(os.path.join('datasets', 'wbwdi.csv'), index_col=1)

'''
Histograms and density plots
Usage: show the distribution of a numerical variable
'''

sns.distplot(wbwdi.LIFEXP.dropna(), kde=False)

# Add title and labels
sns.distplot(wbwdi.LIFEXP.dropna(), kde=False) \
   .set(xlabel='Life expectancy (years)', ylabel='Frequency')

# Compare with the corresponding density plot ('smooth' version of a histogram)
sns.distplot(wbwdi.LIFEXP.dropna())

'''
Scatter plots
Usage: show the relationship between two (or pairs of) numerical variables
'''

# Scatter plot with regression line
sns.lmplot(x='INFMORT', y='LIFEXP', data=wbwdi)
sns.regplot(x='INFMORT', y='LIFEXP', data=wbwdi)

# Scatter plot with marginal histograms
sns.jointplot(x='INFMORT', y='LIFEXP', data=wbwdi)

# 2-D density plot with marginal densities
sns.jointplot(x='INFMORT', y='LIFEXP', data=wbwdi, kind='kde')

# Scatter matrix of three (numerical) variables
sns.pairplot(wbwdi[['INFMORT', 'LIFEXP', 'GNIPCAP']].dropna())

'''
Bar plots
Usage: Show a numerical comparison across categories
'''

# Count the number of countries in each group
wbwdi.Countrygp.value_counts()

# Compare using a bar plot
sns.countplot(x='Countrygp', data=wbwdi)

# Plot average life expectancy by country group
sns.barplot(x='Countrygp', y='LIFEXP', data=wbwdi)

# Plot median life expectancy by country group
sns.barplot(x='Countrygp', y='LIFEXP', data=wbwdi, estimator=np.median)

'''
Box plots
Usage: show quartiles (and outliers) for numerical variables (also across categories)
'''

# Five-number summary (min, Q1, Q2 [median], Q3, max)
wbwdi.LIFEXP.describe()

# Inter-quartile range (IQR): Q3 - Q1
# Outliers: < Q1 - 1.5 * IQR | > Q3 + 1.5 * IQR

# Compare with box plot
sns.boxplot(y='LIFEXP', data=wbwdi)

# Grouped box plots
sns.boxplot(x='Countrygp', y='LIFEXP', data=wbwdi)

'''
Line plots
Usage: show the trend of a numerical variable over time
       (don't use when there's no logical ordering)
'''

simulated_ts = pd.DataFrame({
    't': np.linspace(0, 2 * np.pi)
})

simulated_ts['y'] = np.sin(simulated_ts.t / 4) + 2 * np.cos(simulated_ts.t)

sns.pointplot(x='t', y='y', data=simulated_ts, markers='')

'''
Saving plots to disk
'''

sns.distplot(wbwdi.LIFEXP.dropna(), kde=False) \
   .set(xlabel='Life expectancy (years)', ylabel='Frequency') \

plt.savefig('life_expectancy_hist.pdf')

