from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from math import sqrt
import numpy as np


MAX_ITERATIONS = 1000


def kmeans(data, clusters_number):
    """Kmeans algorith implementation."""
    print "INFO: Clustering."
    # Randomizing centroids.
    centroids = randomize_centroids(data, clusters_number)
    # Initializing previeous_centroids
    previeous_centroids = [[] for i in range(clusters_number)]

    itterations = 0
    while itterations < MAX_ITERATIONS:
        itterations += 1

        # Create empty clusters.
        clusters = [[] for i in range(clusters_number)]
        # Arrange data to clusters.
        assign_to_clusters(clusters, data, centroids)

        if centroids == previeous_centroids:
            print "INFO: Done Clustering."
            return clusters

        # Hard-Copy centroids to previeous_centroids
        previeous_centroids = centroids[:]

        # Re-Calculate centroids using the mean value of all the data that a cluster contains.
        for index, cluster in enumerate(clusters):
            centroids[index] = np.mean(cluster, axis=0).tolist()


    print "INFO: Clustering Timeout."
    return clusters


def randomize_centroids(data, clusters_number):
    """Randomizes cluster_number points. """
    centroids = []
    for i in range(clusters_number):
        # Randomize centroids for initial state.
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return centroids

def prepare_data(dataset):
    """Preprocess data and returns a list of vectors."""
    print "INFO: Preparing Data."
    vectorizer = CountVectorizer(stop_words='english') # stop_words remove irrelevant english words
    svd = TruncatedSVD(n_components=2) # n_components normaly equal to 40 (2 for plotting)
    content = dataset['Content']
    transformer=TfidfTransformer()
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', transformer),
        ('svd', svd)
    ])
    X = pipeline.fit_transform(content)
    print "INFO: Data pipelined."
    return X

def points_to_category(points, data, df):
    """Maps given set of points to original category."""
    category=df.Category
    categories = []

    for point in points:
        categories.append(category[data.tolist().index(point.tolist())])

    return categories

def cosine_similarity(vector1, vector2):
    """Calculates cosine similarity of 2 given vectors """
    xx, xy, yy = 0, 0, 0
    for i in range(len(vector1)):
        x = vector1[i]
        y = vector2[i]

        xx += x * x
        xy += x * y
        yy += y * y
    try:
        return [xy / (sqrt(xx * yy))]
    except ZeroDivisionError:
        return [0]


def assign_to_clusters(clusters, data, centroids):
    """Assigns points to clusters using cosine similarity algorithm."""

    for i in range(len(data.tolist())):
        # Calculates points similarity to all centers
        similarity = [x for y in centroids for x in cosine_similarity(y,data[i].tolist())]
        # Assigns to the most similar (nearest) cluster.
        clusters[similarity.index(max(similarity))].append(data[i])
    # If any cluster is empty then assign one point
    # from data set randomly so as to not have empty
    # clusters and 0 means.
    for cluster in clusters:
         if not cluster:
             cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
