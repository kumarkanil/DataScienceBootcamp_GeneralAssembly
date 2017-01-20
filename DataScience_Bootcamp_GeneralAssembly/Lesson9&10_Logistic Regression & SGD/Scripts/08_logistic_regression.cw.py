#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 8: Logistic regression using StatsModels

* Odds and odds ratios
* Logistic regression
'''

import os

import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

%matplotlib qt

# Read in the Crime Survey for England and Wales 2013-2014 data
csew = pd.read_csv(os.path.join('datasets', 'csew2013.csv'), index_col=0)

# Define predictors and response
predictors = ['sex', 'age', 'walkdark']
response = 'bcsvictim'

all_vars = predictors + [response]

# Recode missing values in 'walkdark'
csew.where(csew['walkdark'] != 8, inplace=True)

# Remove missing values
csew.dropna(subset=all_vars, inplace=True)

# Convert categorical variables to the 'category' data type
csew.sex = csew.sex.astype('category')
csew.walkdark = csew.walkdark.astype('category')

# Explore data visually
sns.boxplot(x='bcsvictim', y='age', data=csew)

'''
Odds and odds ratios
'''

# Probability of having experienced crime by sex
p_men = csew[csew.sex == 1].bcsvictim.mean()
p_women = csew[csew.sex == 2].bcsvictim.mean()

# Corresponding odds
odds_men = p_men / (1 - p_men)
odds_women = p_women / (1 - p_women)

# Alternatively…
odds_men = csew[csew.sex == 1].bcsvictim.sum() /\
           (1 - csew[csew.sex == 1].bcsvictim).sum()
odds_women = csew[csew.sex == 2].bcsvictim.sum() /\
             (1 - csew[csew.sex == 2].bcsvictim).sum()

# Odds ratio of a woman having experienced crime (compared to a man)
odds_women / odds_men

'''
Logistic regression
'''

# Modelling the probability of having experienced crime by sex
# No intercept means there is no reference category
model1 = smf.logit('bcsvictim ~ -1 + sex', data=csew).fit()

model1.summary()
model1.params

# Taking the exponential of the regression coefficients returns the odds
np.exp(model1.params)
odds_men
odds_women

# Including the intercept means one category (sex = 1) acts as reference
model2 = smf.logit('bcsvictim ~ sex', data=csew).fit()

model2.summary()
model2.params

# Taking the exponential of the regression coefficients returns the odds of the
# reference category, and the OR of the outcome in the non-reference category
np.exp(model2.params)
odds_men
odds_women / odds_men

# The odds in the non-reference category are the odds in the reference category
# (intercept) multiplied by the OR in the non-reference category (coefficient)
np.prod(np.exp(model2.params))

# Full model
model3 = smf.logit('bcsvictim ~ sex + age + walkdark', data=csew).fit()
model3.summary()
model3.summary2()

model3.params
model3.pvalues
model3.conf_int()
model3.prsquared

