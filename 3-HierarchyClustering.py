import matplotlib.pyplot as plt
import sys
import numpy as np
import csv
from sklearn import preprocessing
import sklearn.neighbors
from scipy import cluster

if(len(sys.argv) is not 3):
	print("Usage: python3 E3.py <input_file>  <output_file>")
	exit()
csv_file=csv.reader(open(sys.argv[1]))
aux=list(csv_file)

fields=aux[0]
data=aux[1:]
numpyData=list(map(lambda x : list(map(float,x[1:])), aux[1:]))

min_max_scaler = preprocessing.MinMaxScaler()
numpyData = min_max_scaler.fit_transform(numpyData)


#metrics=['euclidean', 'manhattan', 'chebyshev','minkowski','wminkowski','seuclidean','mahalanobis']
#methods=['complete', 'average', 'weighted','single','ward','median','centroid']
metric='euclidean'
method='complete'
threshold=5

dist = sklearn.neighbors.DistanceMetric.get_metric(metric)
matsim = dist.pairwise(numpyData)
avSim = np.average(matsim)

clusters = cluster.hierarchy.linkage(matsim, method = method)
cluster.hierarchy.dendrogram(clusters, color_threshold=threshold)
plt.title(metric+" "+method)
#plt.savefig(me+" "+m, dpi=1000)
plt.show()

clusters=cluster.hierarchy.fcluster(clusters,threshold,criterion='distance')


for i in range(len(data)):
	data[i].append(clusters[i])
