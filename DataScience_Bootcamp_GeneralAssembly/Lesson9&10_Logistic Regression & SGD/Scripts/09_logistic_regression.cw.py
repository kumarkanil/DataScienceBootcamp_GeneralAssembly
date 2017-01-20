#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 9: Logistic regression using scikit-learn

* Logistic regression
* Confusion matrix and performance metrics
* Visualising the effect of covariates
* ROC analysis
* Cross-validation
* Regularisation
* Variable (feature) selection
* Multiple classes
* Stochastic gradient descent
'''

import numpy as np
import pandas as pd

from sklearn import linear_model as lm, metrics, cross_validation as cv,\
                    grid_search, feature_selection, preprocessing

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib qt

# Original publication: http://dx.doi.org/10.1371/journal.pone.0137524
# Dataset: http://dx.doi.org/10.5061/dryad.8jq92

GLIOMA_URL = 'http://datadryad.org/bitstream/handle/10255/dryad.88928/An%20eighteen%20serum%20cytokine%20signature%20for%20discriminating%20glioma%20from%20normal%20healthy%20individuals%20raw%20data.xlsx?sequence=1'

glioma = pd.read_excel(GLIOMA_URL)

# Transpose DataFrame so that measurements are in columns
glioma = glioma.transpose()

# Set first row as column names, then drop it
glioma.columns = glioma.iloc[0]
glioma.columns.name = ''
glioma = glioma.reindex(glioma.index.drop('sample'))

# Extract cytokine measurements
X = glioma.iloc[:,1:].apply(pd.to_numeric, axis=1)

# Apply logarithmic transformation to each measurement
X = X.apply(np.log, axis=1)

# Dichotomise outcome: GBM versus rest
# 
# DA  = Diffuse Astrocytoma (grade II)
# AA  = Anaplastic Astrocytoma (grade III)
# GBM = Glioblastoma Multiforme (grade IV)
y = glioma.Type == 'GBM'

'''
Logistic regression
'''

# Fit the model
# NOTE: By default, sklearn uses L2 regularisation with parameter C (default = 1)
#       This cannot be disabled, but we can set C so big that it has little effect
model1 = lm.LogisticRegression(C=1e50)
model1.fit(X, y)

# Print regression coefficients
model1.intercept_
model1.coef_

# Print odds ratios
np.exp(model1.intercept_)
np.exp(model1.coef_)

'''
Confusion matrix and performance metrics
'''

# Confusion matrix
metrics.confusion_matrix(y, model1.predict(X))

# Classification accuracy
metrics.accuracy_score(y, model1.predict(X))

# Classification report
print(metrics.classification_report(y, model1.predict(X)))

'''
Visualising the effect of covariates
'''

# Scatter plot of IL-6 with regression line
sns.regplot(X['IL6'], y, logistic=True)

# Define a set of evenly spaced IL-6 levels
il6_levels = np.linspace(8, 12, 100)

# Create a new array and fill it with:
# * the IL-6 levels defined above
# * the average observed concentrations of all other cytokines in controls
X2 = np.zeros((il6_levels.size, X.shape[1]))

for i in range(X.shape[1]):
    if i == np.where(X.columns == 'IL6')[0][0]:
        X2[:,i] = il6_levels
    else:
        X2[:,i] = X.ix[y == 0,i].mean()

# Compute predicted probabilities for GBM (y = 1) using the new array
il6_pred_probs = model1.predict_proba(X2)[:,1]

# Plot against IL-6 levels
plt.plot(il6_levels, il6_pred_probs)

'''
ROC analysis
'''

# Compute predicted probabilities for GBM (y = 1)
pred_probs = model1.predict_proba(X)[:,1]

# Confirm that model predictions assume a 50% cut-off value
assert(np.all((pred_probs >= 0.5) == model1.predict(X)))

# Visualise distribution
sns.distplot(pred_probs)

# Define a set of cut-off values where sensitivity and specificity will be computed
cutoffs = np.linspace(0, 1, 1001)

# Define a function to compute specificity
def specificity_score(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm[0,0] / cm[0,:].sum()

# Compute sensitivity and specificity at the cut-off values defined above
sensitivities = np.zeros(cutoffs.size)
specificities = np.zeros(cutoffs.size)
for i, cutoff in enumerate(cutoffs):
    sensitivities[i] = metrics.recall_score(y, pred_probs >= cutoff)
    specificities[i] = specificity_score(y, pred_probs >= cutoff)

# Plot the ROC curve, i.e. sensitivity versus (1 - specificity)
plt.plot(1 - specificities, sensitivities)

# Alternatively…
# (FPR = 1 - specificity; TPR = sensitivity)
fpr, tpr, cutoffs = metrics.roc_curve(y, pred_probs)
plt.plot(fpr, tpr)

# Compute area under the ROC curve (AUC)
metrics.roc_auc_score(y, pred_probs)

'''
Cross-validation
'''

# Define stratified folds
kf = cv.StratifiedKFold(y, n_folds=5, shuffle=True)

# Compute average classification accuracy across folds
accuracies = cv.cross_val_score(lm.LogisticRegression(C=1e50),\
                                X, y, scoring='accuracy', cv=kf)
np.mean(accuracies)

# Compute average AUC across folds
aucs = cv.cross_val_score(lm.LogisticRegression(C=1e50),\
                          X, y, scoring='roc_auc', cv=kf)
np.mean(aucs)

'''
Regularisation
'''

# Determine ‘optimal’ value of C by cross-validation using AUC scoring
# (sklearn uses L2 regularisation by default)
Cs = np.logspace(-4, 4, 10)
gs = grid_search.GridSearchCV(
    estimator=lm.LogisticRegression(),
    param_grid={'C': Cs},
    scoring='roc_auc',
    cv=kf
)
gs.fit(X, y)

gs.best_score_
gs.best_estimator_
gs.grid_scores_

# Plot Cs against AUC scores
plt.plot(np.log(Cs), [ x[1] for x in gs.grid_scores_ ])

# Alternatively…
model2 = lm.LogisticRegressionCV(Cs=10, cv=kf, scoring='roc_auc')
model2.fit(X, y)

plt.plot(np.log(model2.Cs_), model2.scores_[1].mean(axis=0))

np.exp(model2.intercept_)
np.exp(model2.coef_)

# Repeat using L1 regularisation
model3 = lm.LogisticRegressionCV(Cs=10, cv=kf, penalty='l1', scoring='roc_auc',\
                                 solver='liblinear')
model3.fit(X, y)

np.exp(model3.intercept_)
np.exp(model3.coef_)

plt.plot(np.log(model3.Cs_), model3.scores_[1].mean(axis=0))

'''
Variable (feature) selection
'''

# Select only variables with non-zero coefficient in L1-regularised model
idx = np.where(np.abs(model3.coef_[0]) >= 1e-16)[0]

# List selected variables
X.columns[idx]

# Re-fit model using selected variables only
X_selected = X.iloc[:,idx]
model4 = lm.LogisticRegression(C=model3.C_[0])
model4.fit(X_selected, y)

# Plot ROC curve
pred_probs = model4.predict_proba(X_selected)[:,1]
fpr, tpr, cutoffs = metrics.roc_curve(y, pred_probs)
plt.plot(fpr, tpr)

# Compute area under the ROC curve (AUC)
metrics.roc_auc_score(y, pred_probs)

# Recursive Feature Elimination and Cross-Validated selection
# (using value of C found using cross-validation above)
fs = feature_selection.RFECV(lm.LogisticRegression(C=model2.C_[0]),\
                             cv=kf, scoring='roc_auc')
fs.fit(X, y)

# List selected variables
X.columns[fs.support_]

# Re-fit model using selected variables only
X_selected = X.loc[:,fs.support_]
model5 = lm.LogisticRegression(C=model2.C_[0])
model5.fit(X_selected, y)

# Plot ROC curve
pred_probs = model5.predict_proba(X_selected)[:,1]
fpr, tpr, cutoffs = metrics.roc_curve(y, pred_probs)
plt.plot(fpr, tpr)

# Compute area under the ROC curve (AUC)
metrics.roc_auc_score(y, pred_probs)

'''
Multiple classes
'''

# Check the argument `multi_class`:
# * 'ovr' means that binary models are estimated for each class
# * 'multinomial' means that a single multinomial model is estimated

# For example…
model6 = lm.LogisticRegression(C=1e50, solver='lbfgs', multi_class='multinomial')
model6.fit(X, glioma.Type)

model6.classes_
np.exp(model6.intercept_)
np.exp(model6.coef_)

'''
Stochastic gradient descent
'''

# SGD is a very efficient approach to train linear classifiers (including linear
# and logistic regression models) on large-scale and/or sparse datasets
#
# scikit-learn provides:
# * `lm.SGDRegressor` for regression problems
# * `lm.SGDClassifier` for classification problems
#
# Both support L1, L2, and Elastic Net regularisation (with parameters 'alpha'
# and 'l1_ratio' if using Elastic Net)

# SGD is sensitive to scaling of the predictors, so it’s recommended to scale
# the data to [0, 1], [-1, 1], or alternatively to standardise it to mean 0 and
# variance 1, if there’s no ‘intrinsic scale’ already

scaler = preprocessing.StandardScaler()
scaler.fit(X)

scaler.mean_
scaler.scale_

X_scaled = scaler.transform(X)

# According to the scikit-learn documentation, the following is a good guess for
# the number of iterations required to achieve convergence
n_iter = np.ceil(10**6 / X.shape[0])

# As usual, the regularisation parameter 'alpha' can be tuned using
# `grid_search.GridSearchCV`
gs = grid_search.GridSearchCV(
    estimator=lm.SGDClassifier(loss='log', penalty='l2', n_iter=n_iter),
    param_grid={'alpha': 10.0**-np.arange(1, 7)},
    scoring='roc_auc',
    cv=kf
)
gs.fit(X_scaled, y)

gs.best_estimator_

# Before using this model to predict, we'd need to call `scaler.transform` on
# the new data

# We can also put everything together in a pipeline…

sgd_pipeline = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('sgd', lm.SGDClassifier())
])

sgd_pipeline.set_params(
    sgd__loss='log',
    sgd__penalty='l2',
    sgd__n_iter=n_iter
)

gs = grid_search.GridSearchCV(
    estimator=sgd_pipeline,
    param_grid={'sgd__alpha': 10.0**-np.arange(1, 7)},
    scoring='roc_auc',
    cv=kf
)
gs.fit(X, y)

gs.best_estimator_

# Predictions for new samples can now be obtained directly by calling
# `gs.best_estimator_.predict`

