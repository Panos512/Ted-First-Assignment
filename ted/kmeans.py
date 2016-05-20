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
    centroids = randomize_centroids(data, clusters_number)
    previeous_centroids = [[] for i in range(clusters_number)]

    itterations = 0
    while itterations < MAX_ITERATIONS:
        print itterations


        itterations += 1

        clusters = [[] for i in range(clusters_number)]
        assign_to_clusters(clusters, data, centroids)

        if centroids == previeous_centroids:
            print "INFO: Done Clustering."
            return clusters

        previeous_centroids = centroids[:]

        for index, cluster in enumerate(clusters):
            centroids[index] = np.mean(cluster, axis=0).tolist()


    print "INFO: Clustering Timeout."
    return clusters


def randomize_centroids(data, clusters_number):
    """Randomizes cluster_number points. """
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
    """Converts text data to 2d vectors. """
    print "INFO: Preparing Data."
    vectorizer = CountVectorizer(stop_words='english')
    svd = TruncatedSVD(n_components=40)
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
    """Maps given set of 2d points to original category."""
    # FIXME: le = preprocessing.LabelEncoder() could be faster than saving category string.
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
        return [0] # FIXME: Should we return 0 here? Or we should never end up here?
        print "ZE"
        pass


def assign_to_clusters(clusters, data, centroids):
    """Assigns points to clusters using cosine similarity algorithm."""
    #from sklearn.metrics.pairwise import cosine_similarity as cs #FIXME: This gives warnings. But can we use it anyways?

    for i in range(len(data.tolist())):
        # Calculates points similarity to all centers
        similarity = [x for y in centroids for x in cosine_similarity(y,data[i].tolist())]
        # clusters[distances.index(min(distances))].append(data[i]) # FIXME: Should we save the data or a pointer? (index>>)
        # Assigns to the most similar (nearest) cluster.
        clusters[similarity.index(max(similarity))].append(data[i])
    # FIXME: FIXME: FIXME: Take a look!
    # # If any cluster is empty then assign one point
    # # from data set randomly so as to not have empty
    # # clusters and 0 means.
    for cluster in clusters:
         if not cluster:
             cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist()) # FIXME: HUGE copy paste (only this line)
