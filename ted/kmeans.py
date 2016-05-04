from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from math import sqrt
import numpy as np


MAX_ITERATIONS = 1000


def kmeans(data, clusters_number):
    centroids = randomize_centroids(data, clusters_number)
    previeous_centroids = [[] for i in range(clusters_number)]

    itterations = 0
    while itterations < MAX_ITERATIONS: #not shouldstop(centroids, previeous_centroids, itterations):
        if (itterations % 100) == 0:
            print itterations
        itterations += 1

        clusters = [[] for i in range(clusters_number)]
        assign_to_clusters(clusters, data, centroids)

        #previeous_centroids = centroids # FIXME: This is right. Right?
        if centroids == previeous_centroids:
            return clusters

        previeous_centroids = centroids[:]

        for index, cluster in enumerate(clusters):
            centroids[index] = np.mean(cluster, axis=0).tolist()



    return clusters


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
    vectorizer = CountVectorizer(stop_words='english') # FIXME: 12266 lines after fit_transform instead of 61330
    svd = TruncatedSVD(n_components=2)
    content = dataset['Content']
    transformer=TfidfTransformer()
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', transformer),
        ('svd', svd)
    ])
    X = pipeline.fit_transform(content)
    return X


def cosine_similarity(vector1, vector2):
    xx, xy, yy = 0, 0, 0
    for i in range(len(vector1)):
        x = vector1[i]
        y = vector2[i]

        xx += x * x
        xy += x * y
        yy += y * y
    try:
        return xy / (sqrt(xx * yy))
    except ZeroDivisionError:
        #return 0
        pass


def assign_to_clusters(clusters, data, centroids):
    for i in range(len(data.tolist())):

        distances = [x for y in centroids for x in [cosine_similarity(y,data[i].tolist())]]
        clusters[distances.index(min(distances))].append(data[i]) # FIXME: Should we save the data or a pointer? (index>>)

    # FIXME: FIXME: FIXME: Take a look!
    # # If any cluster is empty then assign one point
    # # from data set randomly so as to not have empty
    # # clusters and 0 means.
    for cluster in clusters:
         if not cluster:
             cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist()) # FIXME: HUGE copy paste (only this line)

from kmeans import prepare_data, kmeans
from tools import open_csv

df = open_csv()

x = kmeans(data=prepare_data(df), clusters_number=2)

import ipdb; ipdb.set_trace()
