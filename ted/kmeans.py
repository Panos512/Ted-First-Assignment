from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

import numpy as np


MAX_ITERATIONS = 1000


def kmeans(data, clusters_number):
    centroids = randomize_centroids(data, clusters_number)
    import ipdb; ipdb.set_trace()


def randomize_centroids(data, clusters_number):
    centroids = []
    for i in range(clusters_number):
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return centroids


def shouldstop(current_centroids, previeous_centroids, number_of_iterations):
    """ Decides when K-Means algorithm should stop recalculating centroids sets."""
    if number_of_iterations > MAX_ITERATIONS:
        return True
    return current_centroids == previeous_centroids


def prepare_data(dataset):
    vectorizer = CountVectorizer(stop_words='english')
    svd = TruncatedSVD(n_components=2)
    content = dataset['Content']

    pipeline = Pipeline([
        ('vect', vectorizer),
        ('svd', svd)
    ])
    X = pipeline.fit_transform(content)
    return X
