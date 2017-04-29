import scipy

from collections import Counter

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, metrics, cross_validation
from tools import open_csv

from nltk import PorterStemmer
from nltk.tokenize import RegexpTokenizer


class MyKNN():

    def __init__(self, x_train, y_train, K):
        self.x_train = x_train
        self.y_train = y_train
        self.K = K

    @staticmethod
    def most_common(lst):
        return max(set(lst), key=lst.count)

    def predict(self, points):
        results = []
        for point in points:
            dist = scipy.spatial.distance.cdist(self.x_train, [point])
            indexes = np.argsort(dist[:, 0])[:self.K]
            result = []
            for index in indexes:
                result.append(self.y_train[index])
            results.append(self.most_common(result))

        return results

    def score(self, X_test, y_test):
        result = self.predict(X_test)

        validation_list = []
        for i, j in zip(result, y_test):
            if i == j:
                validation_list.append(1)
            else:
                validation_list.append(0)

        c = Counter(validation_list)
        count = dict([(i, c[i] / float(len(validation_list)) * 100.0) for i in c])
        return count[1]



def preprocess_data(df):
    """Preprocesses the data returning a stemmed and regexped-tokenized represantation of the input."""
    x_train = df['Content']

    X = Preprocessor().fit(x_train)
    return X

class Preprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor that runs regexp tokenizing and stemming."""

    def process(self, sentence):
        return ' '.join(self.stem(word) for word in self.tokenize(sentence))

    def __init__(self):
        self.stem = PorterStemmer().stem
        self.tokenize = RegexpTokenizer(r"\b\w+\b").tokenize

    def fit(self, X, y=None):
        return [self.process(sentence) for sentence in X]

    def transform(self, X, y=None):
        return self


def pipeline_data(X, df):
    """Converts string data to a list of vectors."""
    # from sklearn.feature_extraction.text import TfidfVectorizer
    pipeline = Pipeline([
        ('vec', CountVectorizer(max_features=4096, stop_words='english')),
        ('transformer', TfidfTransformer()),
        ('svd', TruncatedSVD(n_components=40))
    ])
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(df["Category"])

    X = pipeline.fit_transform(X, y_train)
    return X, y_train, le

df = open_csv()
X_Original = preprocess_data(df)
X = X_Original

X, y_train, le = pipeline_data(X, df)


categorys_map={
'Politics': 0,
'Business': 1,
'Film': 2,
'Technology': 3,
'Football': 4
}
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
              X, y_train, test_size=0.4, random_state=0)

knn = MyKNN(X_train, y_train, 3)

print knn.score(X_test, y_test)

