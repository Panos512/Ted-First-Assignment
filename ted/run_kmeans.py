from kmeans import prepare_data, kmeans, points_to_category
from tools import open_csv
import pandas as pd

df = open_csv()
data = prepare_data(df)
x = kmeans(data=data, clusters_number=5)

for index, cluster in enumerate(x):
    print "Cluster " + str(index)
    res = points_to_category(points=cluster, data=data, df=df)
    print pd.value_counts(res, normalize=True)
