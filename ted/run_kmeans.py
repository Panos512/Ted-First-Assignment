from kmeans import prepare_data, kmeans, points_to_category
from tools import open_csv
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--plot', action='store_true')
args = parser.parse_args()

if not args.plot:
    print "Run with -p to plot clusters."

df = open_csv()
data = prepare_data(df)



x = kmeans(data=data, clusters_number=5)

color={
0:'red',
1:'green',
2:'blue',
3:'black',
4:'yellow'}


for index, cluster in enumerate(x):
    print "Cluster " + str(index)
    if args.plot:
        for point in cluster:
            plt.scatter(point[0],point[1],color = color[index])
    res = points_to_category(points=cluster, data=data, df=df)
    print pd.value_counts(res, normalize=True)
plt.show()
