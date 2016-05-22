import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, metrics, cross_validation

from nltk import PorterStemmer
from nltk.tokenize import RegexpTokenizer


class Preprocessor(BaseEstimator, TransformerMixin):
    """Insert docstring."""

    def process(self, sentence):
        return ' '.join(self.stem(word) for word in self.tokenize(sentence))

    def __init__(self):
        self.stem = PorterStemmer().stem
        self.tokenize = RegexpTokenizer(r"\b\w+\b").tokenize

    def fit(self, X, y=None):
        return [self.process(sentence) for sentence in X]

    def transform(self, X, y=None):
        return self


def calculate_accuracy(X, y_train, algorithm):
    """Insert docstring."""
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y_train, test_size=0.4, random_state=0)
    pipeline = Pipeline([
        ('vec', CountVectorizer(max_features=4096, stop_words='english')),
        ('transformer', TfidfTransformer()),
        ('svd', TruncatedSVD(n_components=40)),
        ('clf', algorithm)
    ])
    pipeline.fit(X_train, y_train)
    print pipeline.score(X_test, y_test)


def preprocess_data(df):
    """Insert docstring."""
    x_train = df['Content']

    X = Preprocessor().fit(x_train)
    return X


def pipeline_data(X, df):
    """Insert docstring."""
    # from sklearn.feature_extraction.text import TfidfVectorizer
    pipeline = Pipeline([
        ('vec', CountVectorizer(max_features=4096, stop_words='english')),
        ('transformer', TfidfTransformer()),
        #('vec', TfidfVectorizer(min_df=3, max_df=.95, stop_words='english')),
        ('svd', TruncatedSVD(n_components=40))
        #('pr', preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True))
    ])
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(df["Category"])

    X = pipeline.fit_transform(X, y_train)
    return X, y_train, le


def create_classifier(df, X, X_Original, y_train, le, algorithm):
    """Insert docstring."""
    calculate_accuracy(X_Original, y_train, algorithm)
    algorithm.fit(X, y_train)
    predicted = algorithm.predict(X)
    print(metrics.classification_report(
        df['Category'], le.inverse_transform(predicted)))
    return algorithm


def classify_data(algorithm):
    """Insert docstring."""
    path = '/Users/Panos/.virtualenvs/ted/src/ted/data_sets/test_set.csv'
    dft = pd.read_csv(path, sep='\t', encoding='utf-8')
    X = preprocess_data(dft)
    # pipeline = Pipeline([
    #     ('vec', CountVectorizer(max_features=4096, stop_words='english')),
    #     ('transformer', TfidfTransformer()),
    #     ('svd', TruncatedSVD(n_components=40))
    # ])
    #from sklearn.feature_extraction.text import TfidfVectorizer
    pipeline = Pipeline([
        ('vec', CountVectorizer(max_features=4096, stop_words='english')),
        ('transformer', TfidfTransformer()),
        #('vec', TfidfVectorizer(min_df=3, max_df=.95, stop_words='english')),
        ('svd', TruncatedSVD(n_components=40))
        #('pr', preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True))
    ])
    X = pipeline.fit_transform(X)
    predicted = algorithm.predict(X)
    headers = ['id', 'Category']
    results = pd.DataFrame(
        zip(dft.Id, le.inverse_transform(predicted)), columns=headers)
    results.to_csv('./ted/outputs/testSet_categories.csv', index=False)

path = '/Users/Panos/.virtualenvs/ted/src/ted/data_sets/train_set.csv'
df = pd.read_csv(path, sep='\t', encoding='utf-8')
X_Original = preprocess_data(df)
X = X_Original
X, y_train, le = pipeline_data(X, df)


# from sklearn.naive_bayes import MultinomialNB
# print "INFO: Multinomial Classification"
# clf = MultinomialNB()
# import ipdb; ipdb.set_trace()
# algorithm = create_classifier(df, X, X_Original, y_train, le, clf)
# ipdb.set_trace()
# classify_data(algorithm)
# ipdb.set_trace()

from sklearn.naive_bayes import GaussianNB
print "INFO: GaussianNB Classification"
clf = GaussianNB()
algorithm = create_classifier(df, X, X_Original, y_train, le, clf)
classify_data(algorithm)

from sklearn.naive_bayes import BernoulliNB
print "INFO: Binomial BernoulliNB Classification"
clf = BernoulliNB()
algorithm = create_classifier(df, X, X_Original, y_train, le, clf)
classify_data(algorithm)

from sklearn.neighbors import KNeighborsClassifier
print "INFO: KNeighbors Classification"
clf = KNeighborsClassifier(n_neighbors=3)
algorithm = create_classifier(df, X, X_Original, y_train, le, clf)
classify_data(algorithm)


from sklearn.ensemble import RandomForestClassifier
print "INFO: Random Forest Classification"
clf = RandomForestClassifier(n_estimators=10)
algorithm = create_classifier(df, X, X_Original, y_train, le, clf)
classify_data(algorithm)

from sklearn.svm import SVC
print "INFO: SVM Classification"
clf = SVC()
algorithm = create_classifier(df, X, X_Original, y_train, le, clf)
classify_data(algorithm)



minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
