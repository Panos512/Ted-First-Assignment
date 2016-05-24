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


def calculate_accuracy(X, y_train, algorithm):
    """Calculates the accuracy of a given classification algorithm."""
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y_train, test_size=0.4, random_state=0)
    pipeline = Pipeline([
        ('vec', CountVectorizer(max_features=4096, stop_words='english')),
        ('transformer', TfidfTransformer()),
        ('svd', TruncatedSVD(n_components=40)),
        ('clf', algorithm)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline.score(X_test, y_test)


def calculate_accuracy_naive_bayes(X, y_train, algorithm):
    """Calculates the accuracy of a given naive-bayes classification algorithm.(not using TruncatedSVD for naive-bayes)"""
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y_train, test_size=0.4, random_state=0)
    pipeline = Pipeline([
        ('vec', CountVectorizer(max_features=4096, stop_words='english')),
        ('transformer', TfidfTransformer()),
        ('clf', algorithm)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline.score(X_test, y_test)


def calculate_score(X, y_train, algorithm):
    """Calculates the scores of a given classification algorithm."""
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y_train, test_size=0.4, random_state=0)
    pipeline = Pipeline([
        ('vec', CountVectorizer(max_features=4096, stop_words='english')),
        ('transformer', TfidfTransformer()),
        ('svd', TruncatedSVD(n_components=40)),
        ('clf', algorithm)
    ])
    y = y_train
    y_train = preprocessing.label_binarize(y_train, classes=[0, 1, 2, 3])
    print cross_validation.cross_val_score(pipeline, X_train, y_train, cv=3, scoring='roc_auc')

def calculate_score_naive_bayes(X, y_train, algorithm):
    """Calculates the scores of a given Naive-Bayes classification algorithm (not using TruncatedSVD)."""
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y_train, test_size=0.4, random_state=0)
    pipeline = Pipeline([
        ('vec', CountVectorizer(max_features=4096, stop_words='english')),
        ('transformer', TfidfTransformer()),
        ('clf', algorithm)
    ])
    y = y_train
    y_train = preprocessing.label_binarize(y_train, classes=[0, 1, 2, 3])
    import ipdb; ipdb.set_trace()
    print cross_validation.cross_val_score(pipeline, X_train, y_train, cv=3, scoring='roc_auc')


def preprocess_data(df):
    """Preprocesses the data returning a stemmed and regexped-tokenized represantation of the input."""
    x_train = df['Content']

    X = Preprocessor().fit(x_train)
    return X


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


def pipeline_data_naive_bayes(X, df):
    """Converts string data to a list of vectors. (not using TruncatedSVD for naive-bayes)"""
    pipeline = Pipeline([
        ('vec', CountVectorizer(max_features=4096, stop_words='english')),
        ('transformer', TfidfTransformer())
    ])
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(df["Category"])

    X = pipeline.fit_transform(X, y_train)
    return X, y_train, le


def create_classifier(df, X, X_Original, y_train, le, algorithm, multinomial=0):
    """Creates a classifier based on the algorithm and tests accuracy."""
    if multinomial:
        accuracy = calculate_accuracy_naive_bayes(X_Original, y_train, algorithm)
        #calculate_score_naive_bayes(X_Original, y_train, algorithm)
    else:
        accuracy = calculate_accuracy(X_Original, y_train, algorithm)
        #calculate_score(X_Original, y_train, algorithm)
    algorithm.fit(X, y_train)
    predicted = algorithm.predict(X)
    print(metrics.classification_report(
        df['Category'], le.inverse_transform(predicted)))
    return algorithm,accuracy


def classify_data(algorithm):
    """Classifies test_data based on the classification algorithm given in the input."""
    dft = open_csv(relative_path='./data_sets/test_set.csv')
    X = preprocess_data(dft)
    pipeline = Pipeline([
        ('vec', CountVectorizer(max_features=4096, stop_words='english')),
        ('transformer', TfidfTransformer()),
        ('svd', TruncatedSVD(n_components=40))
    ])
    X = pipeline.fit_transform(X)
    predicted = algorithm.predict(X)
    headers = ['id', 'Category']
    results = pd.DataFrame(
        zip(dft.Id, le.inverse_transform(predicted)), columns=headers)
    results.to_csv('./ted/outputs/testSet_categories.csv', index=False)


def classify_data_naive_bayes(algorithm):
    """Classifies test_data based on the naive-bayes classification algorithm given in the input.  (not using TruncatedSVD for naive-bayes)"""
    dft = open_csv(relative_path='./data_sets/test_set.csv')
    X = preprocess_data(dft)
    pipeline = Pipeline([
        ('vec', CountVectorizer(max_features=4096, stop_words='english')),
        ('transformer', TfidfTransformer())
    ])
    X = pipeline.fit_transform(X)
    predicted = algorithm.predict(X)
    headers = ['id', 'Category']
    results = pd.DataFrame(
        zip(dft.Id, le.inverse_transform(predicted)), columns=headers)
    results.to_csv('./ted/outputs/testSet_categories.csv', index=False)


# General pre-processing for all the algorithms
df = open_csv()
X_Original = preprocess_data(df)
X = X_Original
accuracy_metrics = []

# Runing naive-bayes algorithms without using TruncatedSVD
X, y_train, le = pipeline_data_naive_bayes(X, df)
from sklearn.naive_bayes import MultinomialNB
print "INFO: MultinomialNB Classification"
clf = MultinomialNB()
algorithm, accuracy = create_classifier(
    df, X, X_Original, y_train, le, clf, multinomial=1)
classify_data_naive_bayes(algorithm)
accuracy_metrics.append(accuracy)

from sklearn.naive_bayes import BernoulliNB
print "INFO: Binomial BernoulliNB Classification"
clf = BernoulliNB()
algorithm, accuracy = create_classifier(df, X, X_Original, y_train, le, clf)
classify_data_naive_bayes(algorithm)
accuracy_metrics.append(accuracy)

# Re-processes data using TruncatedSVD for the rest algorithms
X = X_Original
# FIXME: After scorings apofasisame gia naive_bayes na mhn kanoume truncate
X, y_train, le = pipeline_data(X, df)


# Runing the non naive_bayes algorithms

from sklearn.neighbors import KNeighborsClassifier
print "INFO: KNeighbors Classification"
clf = KNeighborsClassifier(n_neighbors=3)
algorithm, accuracy = create_classifier(df, X, X_Original, y_train, le, clf)
classify_data(algorithm)
accuracy_metrics.append(accuracy)


from sklearn.ensemble import RandomForestClassifier
print "INFO: Random Forest Classification"
clf = RandomForestClassifier(n_estimators=10)
algorithm, accuracy = create_classifier(df, X, X_Original, y_train, le, clf)
classify_data(algorithm)
accuracy_metrics.append(accuracy)


from sklearn.svm import SVC
print "INFO: SVM Classification"
clf = SVC()
algorithm, accuracy = create_classifier(df, X, X_Original, y_train, le, clf)
classify_data(algorithm)
accuracy_metrics.append(accuracy)


# Generating scoring csv
headers = ['MultinomialNB', 'BernoulliNB', 'KNeighbors', 'RandomForestClassifier', 'SVM']
accuracy_metrics = [accuracy_metrics]
results = pd.DataFrame(accuracy_metrics, columns=headers)
results.to_csv('./ted/outputs/EvaluationMetric_10fold.csv', index=False)
