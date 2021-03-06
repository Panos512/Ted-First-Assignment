from os import path

import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

#Read Data
def open_csv(relative_path='../data_sets/train_set.csv'):
    d = path.dirname(__file__)
    train_set_path = path.join(d, relative_path)
    df = pd.read_csv(train_set_path, sep='\t')
    return df

df = open_csv()
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train=le.transform(df["Category"])
X_train=df['Content']
vectorizer=CountVectorizer(stop_words='english')
transformer=TfidfTransformer()
svd=TruncatedSVD(n_components=10)
clf=SGDClassifier()
pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', transformer),
    ('svd',svd),
    ('clf', clf)
])
#Simple Pipeline Fit
pipeline.fit(X_train,Y_train)
#Predict the train set
predicted=pipeline.predict(X_train)
print(metrics.classification_report(df['Category'], le.inverse_transform(predicted)))
