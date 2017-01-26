import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn import metrics

all_data = pd.read_csv('super_clean.csv', sep=',',encoding='Latin1')

print(all_data['price'].size)

numpyData=np.array([all_data['price'],all_data['kilometer'],all_data['antiguedad']]).T

min_max_scaler = preprocessing.MinMaxScaler()
numpyData = min_max_scaler.fit_transform(numpyData)

db=DBSCAN(eps=0.015,min_samples=8,metric='euclidean')
ydb=db.fit_predict(numpyData)
all_data['cluster']=ydb
nClusters=max(ydb)
print("Clusters "+str(nClusters))

clusters=[all_data[all_data['cluster'].isin([i])] for i in range(nClusters+1)]

print(clusters[0])

df=pd.DataFrame(columns=('cluster','ncars','meanPowerPS','diferentModels','diferentBrand','diferentCarType','monthsMean','priceMean','kilometerMean'))

for i in range(nClusters+1):
	c=clusters[i]
	count=c['cluster'].count()
	meanPowerPS=c['powerPS'].mean()
	diferentModels=len(c['model'].unique())
	diferentBrand=len(c['brand'].unique())
	diferentCarType=len(c['vehicleType'].unique())
	monthsMean=c['antiguedad'].mean()
	priceMean=c['price'].mean()
	kilometerMean=c['kilometer'].mean()
	df.loc[i]=[i,count,meanPowerPS,diferentModels,diferentBrand,diferentCarType,monthsMean,priceMean,kilometerMean]
print (df)
