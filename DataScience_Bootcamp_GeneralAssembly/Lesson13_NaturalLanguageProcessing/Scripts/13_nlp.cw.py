#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 13: Natural language processing

* Pre-processing using spaCy
* Bag-of-words and random forests
* LDA using gensim
* word2vec using gensim
'''

import numpy as np
import pandas as pd

from sklearn import cross_validation as cv, feature_extraction as fe, ensemble

from scipy.sparse import hstack

import spacy
from spacy.en import English

from gensim.matutils import Sparse2Corpus
from gensim.models import LdaModel, Word2Vec

H2020_URL = 'http://cordis.europa.eu/data/cordis-h2020projects.csv'

'''
Pre-processing using spaCy
'''

# Initialise spaCy
en = English()

# Parse example sentence
parsed = en('The serpentine syntax of legal language is often used to' +
            ' obfuscate meaning and confuse those outside the law.')

# Extract information
for word in parsed:
    print("{:15}{:15}{:15}{:15}{:15}".format(word.text, word.pos_, word.dep_, \
                                             word.lemma_, word.head.lemma_))

'''
Bag-of-words and random forests
'''

# Read in the H2020 dataset
h2020 = pd.read_csv(H2020_URL, sep=';')

# Convert 'totalCost' and 'ecMaxContribution' to numeric
h2020['totalCost'] = pd.to_numeric(h2020.totalCost.map(lambda x: x.replace(',', '.')))
h2020['ecMaxContribution'] = pd.to_numeric(h2020.ecMaxContribution.map(lambda x: x.replace(',', '.')))

# Keep only signed contracts
h2020 = h2020[h2020.status == 'SIGNED']

# Create a new variable representing whether the project was fully funded by the
# European Commission
h2020['fully_funded'] = ~(h2020.ecMaxContribution < h2020.totalCost)

# Count words and 2-grams (combinations of two words) in the 'objective',
# keeping only those that occur at least 5 times
vectorizer = fe.text.CountVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=5
)

# Prepare the data for use in scikit-learn
X = vectorizer.fit_transform(h2020.objective)
y = h2020.fully_funded.astype('int')

# Include total project cost and coordinator country (using the UK as reference)
country_dummies = pd.get_dummies(h2020.coordinatorCountry).drop('UK', axis=1)
X = hstack([X, np.asmatrix(h2020.totalCost).T, country_dummies])

# Train a random forest with 20 decision trees
rf1 = ensemble.RandomForestClassifier(n_estimators=20)
rf1.fit(X, y)

# Define stratified folds for cross-validation
kf = cv.StratifiedKFold(y, n_folds=10, shuffle=True)

# Compute average AUC across folds
aucs = cv.cross_val_score(rf1, X, y, scoring='roc_auc', cv=kf)
np.mean(aucs)

# Extract variable importances and sort in descending order
importances = pd.DataFrame({
    'variable': vectorizer.get_feature_names() + ['totalCost'] + list(country_dummies.columns),
    'importance': rf1.feature_importances_
})
importances.sort_values('importance', ascending=False, inplace=True)
importances.head(10)

# Compute tf–idf
# (alternatively use `TfidfTransformer` on the output of `CountVectorizer`)
vectorizer = fe.text.TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=5
)

# Prepare the data for use in scikit-learn
X_tfidf = vectorizer.fit_transform(h2020.objective)

# Include total project cost and coordinator country
X_tfidf = hstack([X_tfidf, np.asmatrix(h2020.totalCost).T, country_dummies])

# Train a random forest with 20 decision trees
rf2 = ensemble.RandomForestClassifier(n_estimators=20)
rf2.fit(X_tfidf, y)

# Compute average AUC across folds
aucs = cv.cross_val_score(rf2, X_tfidf, y, scoring='roc_auc', cv=kf)
np.mean(aucs)

# Extract variable importances and sort in descending order
importances = pd.DataFrame({
    'variable': vectorizer.get_feature_names() + ['totalCost'] + list(country_dummies.columns),
    'importance': rf2.feature_importances_
})
importances.sort_values('importance', ascending=False, inplace=True)
importances.head(10)

'''
LDA using gensim
'''

# Count words in the 'objective', keeping only those that occur at least 5 times
vectorizer = fe.text.CountVectorizer(
    stop_words='english',
    min_df=5
)
X = vectorizer.fit_transform(h2020.objective)

# Convert to gensim format
corpus = Sparse2Corpus(X, documents_columns=False)

# Create mapping from word IDs (integers) to words (strings)
id2word = dict(enumerate(vectorizer.get_feature_names()))

# Fit LDA model with 10 topics
lda = LdaModel(corpus=corpus, num_topics=10, id2word=id2word)

# Show top 5 words for each of the 10 topics
lda.show_topics(num_topics=10, num_words=5)

'''
word2vec using gensim
'''

# Convert adjectives and verbs to corresponding lemmas using spaCy
objectives = [ \
    [ x.lemma_ if x.pos == spacy.parts_of_speech.ADJ or \
                  x.pos == spacy.parts_of_speech.VERB \
      else x.text \
      for x in en(text) ] \
    for text in h2020.objective ]

# Fit word2vec model
w2c = Word2Vec(sentences=objectives, size=100, window=5, min_count=5)

# Which words are most similar to 'UK'?
w2c.most_similar('UK')

# Which words are most similar to 'UK' but not related to 'France'?
w2c.most_similar(positive=['UK'], negative=['France'])

# Which word doesn’t go with the others?
w2c.doesnt_match(['Italy', 'Japan', 'France', 'UK'])

