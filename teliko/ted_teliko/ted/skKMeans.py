from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from tools import open_csv


df = open_csv()
content = df['Content']

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(content)

true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)


results = []
headers = ['Politics', 'Business', 'Film', 'Technology', 'Football']

for j in range(true_k):
    indices = [df['Category'][i] for i, x in enumerate(model.labels_) if x == j]
    c = Counter(indices)
    percentages = [(i, c[i] / float(len(indices)) * 100.0) for i in c]
    print percentages
    result = []
    result.append('Cluster ' + str(j))

    percentages = dict(percentages)
    for category in headers:
        if category not in percentages:
            percentages[category] = 0.0
    print percentages
    result = result + [percentages['Politics'], percentages['Business'], percentages['Film'], percentages['Technology'], percentages['Football']]
    results.append(result)

# import ipdb; ipdb.set_trace()

headers = ['Cluster', 'Politics', 'Business', 'Film', 'Technology', 'Football']

results = pd.DataFrame(results, columns=headers)
results.to_csv('./outputs/clustering_KMeans_sklearn.csv', index=False)



