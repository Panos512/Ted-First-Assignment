import pandas as pd

path = '/Users/Panos/.virtualenvs/ted/src/ted/data_sets/train_set.csv'
df = pd.read_csv(path, sep='\t', encoding='utf-8')

import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, metrics, cross_validation

from nltk import PorterStemmer
from nltk.tokenize import RegexpTokenizer


class Preprocessor(BaseEstimator, TransformerMixin):

    def process(self, sentence):
        return ' '.join(self.stem(word) for word in self.tokenize(sentence))

    def __init__(self):
        self.stem = PorterStemmer().stem
        self.tokenize = RegexpTokenizer(r"\b\w+\b").tokenize

    def fit(self, X, y=None):
        return [self.process(sentence) for sentence in X]

    def transform(self, X, y=None):
        return self


le = preprocessing.LabelEncoder()

y_train_original = le.fit_transform(df["Category"])
X_train_original = df['Content']

X = Preprocessor().fit(X_train_original)
calculate_accuracy(X, y_train)

pipeline = Pipeline([
    ('vec', CountVectorizer(max_features=4096, stop_words='english')),
    ('transformer', TfidfTransformer()),
    ('svd', TruncatedSVD(n_components=40)),
    ('clf', GaussianNB())
])


pipeline.fit(X, y_train_original)


predicted = pipeline.predict(X)

print(metrics.classification_report(
    df['Category'], le.inverse_transform(predicted)))
path = '/Users/Panos/.virtualenvs/ted/src/ted/data_sets/test_set.csv'
dft = pd.read_csv(path, sep='\t')
X_test = dft['Content']
predicted = pipeline.predict(X_test)
headers = ['id', 'Category']
results = pd.DataFrame(
    zip(dft.Id, le.inverse_transform(predicted)), columns=headers)
results.to_csv('outputs/testSet_categories.csv', index=False)


"""Metrics."""


def calculate_accuracy(X, y_train)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y_train, test_size=0.4, random_state=0)

    pipeline.fit(X_train, y_train)
    pipeline.score(X_test, y_test)
