import matplotlib.pyplot as plt
import sys
import numpy as np
import csv
from sklearn import preprocessing
import sklearn.neighbors
from scipy import cluster

#Arguments management
if(len(sys.argv) is not 3):
	print("Usage: python3 3-HierarchyClustering.py <input_file> <output_file>")
	exit()
#File opening
csv_file=csv.reader(open(sys.argv[1]))
aux=list(csv_file)

fields=aux[0]
data=aux[1:]
numpyData=list(map(lambda x : list(map(float,[x[2]]+x[9:])), aux[1:]))

#Data normalization
min_max_scaler = preprocessing.MinMaxScaler()
numpyData = min_max_scaler.fit_transform(numpyData)

#Hierarchycal Clustering
metric='euclidean'
method='complete'
threshold=5

dist = sklearn.neighbors.DistanceMetric.get_metric(metric)
matsim = dist.pairwise(numpyData)
avSim = np.average(matsim)

clusters = cluster.hierarchy.linkage(matsim, method = method)

#Dendrogram plotting
cluster.hierarchy.dendrogram(clusters, color_threshold=threshold)
plt.title(metric+" "+method)
plt.show()

#Dendrogram cut
clusters=cluster.hierarchy.fcluster(clusters,threshold,criterion='distance')

#Adding the cluster field to the data
for i in range(len(data)):
	data[i].append(clusters[i])

#Plotting sumary
colNCars=fields.index('ncars')
colPrice=fields.index('priceMean')
colKm=fields.index('kilometerMean')
colMonth=fields.index('monthsMean')
colDamages=fields.index('nBrokenCars')

for i in range(min(clusters),max(clusters)+1):
	count=0
	price=0
	months=0
	km=0
	broken=0
	ncars=0
	for d in data:
		if d[-1]==i:
			count+=1
			price+=float(d[colPrice])
			months+=float(d[colMonth])
			km+=float(d[colKm])
			ncars+=float(d[colNCars])
			broken+=float(d[colDamages])*float(d[colNCars])
	print(i,ncars,price/count,km/count,months/count, broken/ncars)


#Writing file
data.insert(0,fields+['group'])
with open(sys.argv[2],'w') as f:
	writer=csv.writer(f,fields+['group'])
	writer.writerows(data)
