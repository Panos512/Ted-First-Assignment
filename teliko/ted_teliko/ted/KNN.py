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

    def n_fold_validation(self, n, X_test, y_test):
        res = 0
        for i in range(n):
            res = res + self.score(X_test, y_test)

        return res/n


# df = open_csv()
# X_Original = preprocess_data(df)
# X = X_Original
#
# X, y_train, le = pipeline_data(X, df)
#
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(
#               X, y_train, test_size=0.4, random_state=0)
#
# knn = MyKNN(X_train, y_train, 3)
#
#
# print knn.n_fold_validation(10, X_test, y_test)

