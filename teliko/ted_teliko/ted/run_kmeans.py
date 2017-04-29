from kmeans import prepare_data, kmeans, points_to_category
from tools import open_csv
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--plot', action='store_true')
args = parser.parse_args()

if not args.plot:
    print "Run with -p to plot clusters."

df = open_csv()
data = prepare_data(df)

if args.plot:
    print "Make sure that TruncatedSVD n_components is set to 2."
    import matplotlib.pyplot as plt
    for point in data:
        plt.scatter(point[0],point[1],color = 'green')
    plt.savefig("./outputs/pre_clustering_data.png")


x = kmeans(data=data, clusters_number=5)


color={
0:'red',
1:'green',
2:'blue',
3:'black',
4:'yellow'}

categorys_map={
'Politics': 0,
'Business': 1,
'Film': 2,
'Technology': 3,
'Football': 4
}

results = []
for index, cluster in enumerate(x):
    if args.plot:
        for point in cluster:
            plt.scatter(point[0],point[1],color = color[index])
    res = points_to_category(points=cluster, data=data, df=df)
    stats = pd.value_counts(res, normalize=True)
    cluster_result = [0 for i in range(5)]
    categories = (stats.index)
    for index, category in enumerate(stats):
        cluster_result[categorys_map[categories[index]]] = category
    results.append(cluster_result)
import ipdb; ipdb.set_trace()
if args.plot:
    plt.savefig("./outputs/clustered_data.png")

headers = ['Cluster', 'Politics', 'Bussiness', 'Film', 'Technology', 'Football']
for index, result in enumerate(results):
    result.insert(0 ,('Cluster '+str(index+1)))

results = pd.DataFrame(results, columns=headers)
results.to_csv('./outputs/clustering_KMeans.csv', index=False)
