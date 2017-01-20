#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 2: Visualisation with matplotlib

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

# Check the documentation for '%matplotlib' to be able to work interactively
%matplotlib qt

# Increase figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

# List available plot styles
plt.style.available

# Change to a different style
plt.style.use('seaborn-dark-palette')

# Read in the World Bank World Development Indicators data
wbwdi = pd.read_csv(os.path.join('datasets', 'wbwdi.csv'), index_col=1)

'''
Histograms and density plots
Usage: show the distribution of a numerical variable
'''

wbwdi.LIFEXP.plot.hist()               # 10 bins by default
wbwdi.LIFEXP.plot(kind='hist')

# Try changing the number of bins
wbwdi.LIFEXP.plot.hist(20)

# Add title and labels
wbwdi.LIFEXP.plot.hist(20, title='Histogram of life expectancy')
plt.xlabel('Life expectancy (years)')
plt.ylabel('Frequency')

# Compare with the corresponding density plot ('smooth' version of a histogram)
wbwdi.LIFEXP.plot.density(xlim=(0, 100))

# Grouped histograms
wbwdi.hist(column='LIFEXP', by='Countrygp')

# Grouped histograms with shared x-axis
wbwdi.hist(column='LIFEXP', by='Countrygp', sharex=True)

# Grouped histograms with shared x- and y-axes
wbwdi.hist(column='LIFEXP', by='Countrygp', sharex=True, sharey=True)

'''
Scatter plots
Usage: show the relationship between two (or pairs of) numerical variables
'''

wbwdi.plot.scatter(x='INFMORT', y='LIFEXP')

# Add transparency
wbwdi.plot.scatter(x='INFMORT', y='LIFEXP', alpha=0.3)

# Vary point colour by GNI per capita
wbwdi.plot.scatter(x='INFMORT', y='LIFEXP', c='GNIPCAP', colormap='Blues')

# Scatter matrix of three (numerical) variables
pd.scatter_matrix(wbwdi[['INFMORT', 'LIFEXP', 'GNIPCAP']], figsize=(10, 8))

'''
Bar plots
Usage: Show a numerical comparison across categories
'''

# Count the number of countries in each group
wbwdi.Countrygp.value_counts()

# Compare using a bar plot (notice the difference when using `sort_index`)
wbwdi.Countrygp.value_counts().plot.bar()
wbwdi.Countrygp.value_counts().sort_index().plot.bar()

# Compute average life expectancy and infant mortality for each country group
wbwdi.groupby('Countrygp')[['LIFEXP', 'INFMORT']].mean()

# Plot side-by-side
wbwdi.groupby('Countrygp')[['LIFEXP', 'INFMORT']].mean().plot.bar()

# Stacked bar plots
wbwdi.groupby('Countrygp')[['INFMORT', 'UND5MORT']].mean().plot.bar(stacked=True)

'''
Box plots
Usage: show quartiles (and outliers) for numerical variables (also across categories)
'''

# Five-number summary (min, Q1, Q2 [median], Q3, max)
wbwdi.LIFEXP.describe()

# Inter-quartile range (IQR): Q3 - Q1
# Outliers: < Q1 - 1.5 * IQR | > Q3 + 1.5 * IQR

# Compare with box plot
wbwdi.LIFEXP.plot.box()

# Include multiple variables
wbwdi[['LIFEXP', 'INFMORT']].plot.box()

# Grouped box plots
wbwdi.boxplot(column=['LIFEXP', 'INFMORT'], by='Countrygp')

'''
Line plots
Usage: show the trend of a numerical variable over time
       (don't use when there's no logical ordering)
'''

simulated_ts = pd.DataFrame({
    't': np.linspace(0, 2 * np.pi)
})

simulated_ts['y'] = np.sin(simulated_ts.t / 4) + 2 * np.cos(simulated_ts.t)

simulated_ts.plot.line(x='t', y='y')

'''
Saving plots to disk
'''

wbwdi.LIFEXP.plot.hist(20, title='Histogram of life expectancy')
plt.xlabel('Life expectancy (years)')
plt.ylabel('Frequency')
plt.savefig('life_expectancy_hist.pdf')

