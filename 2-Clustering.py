#imports
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn import metrics

#Arguments management
if(len(sys.argv) is not 3):
	print("Usage: python3 E3.py <input_file>  <output_file>")
	exit()

#File opening
all_data = pd.read_csv(sys.argv[1], sep=',',encoding='Latin1')

#Printing data size
print(all_data['price'].size)

#Extracting the elements to use for the clustering
numpyData=np.array([all_data['price'],all_data['kilometer'],all_data['antiguedad']]).T

#Normalizing data
min_max_scaler = preprocessing.MinMaxScaler()
numpyData = min_max_scaler.fit_transform(numpyData)

#Clustering
db=DBSCAN(eps=0.015,min_samples=8,metric='euclidean')
ydb=db.fit_predict(numpyData)
all_data['cluster']=ydb
nClusters=max(ydb)
print("Clusters "+str(nClusters))


#Spliting elements of each cluster in diferents dataFrames
clusters=[all_data[all_data['cluster'].isin([i])] for i in range(nClusters+1)]

#Defining new features for the clusters
df=pd.DataFrame(columns=('cluster', 'ncars', 'meanPowerPS', 'diferentModels', 'diferentBrand', 'monthsMean', 'priceMean', 'kilometerMean', 'nBrokenCars', 'nBenzin', 'nCng', 'nDiesel', 'nHybrid', 'nElektro', 'nLpg', 'nAndere', 'nBus', 'nCabrio', 'nCoupe', 'nKleinwagen', 'nSuv', 'nKombi', 'nLimousine'))

for i in range(nClusters+1):
	c=clusters[i]
	count=c['cluster'].count()
	meanPowerPS=c['powerPS'].mean()
	diferentModels=len(c['model'].unique())
	diferentBrand=len(c['brand'].unique())
	monthsMean=c['antiguedad'].mean()
	priceMean=c['price'].mean()
	kilometerMean=c['kilometer'].mean()
	try:
		nBrokenCars=c.groupby('notRepairedDamage').size()['ja']/count
	except KeyError:
		nBrokenCars=0
	#FuelTypes
	types=c.groupby('fuelType').size()
	def getN(fType):
		try:
			return types[fType]/count
		except KeyError:
			return 0
	nFuelTypes=[getN('benzin'),getN('cng'),getN('diesel'),getN('hybrid'),getN('elektro'),getN('lpg')]
	#VehicleTypes
	types=c.groupby('vehicleType').size()
	nVehicleTypes=[getN('andere'),getN('bus'),getN('cabrio'),getN('coupe'),getN('kleinwagen'),getN('suv'),getN('kombi'),getN('limousine')]
	
	df.loc[i]=[i,count,meanPowerPS,diferentModels,diferentBrand,monthsMean,priceMean,kilometerMean,nBrokenCars]+nFuelTypes+nVehicleTypes
csv_file=df.to_csv(index=False)

#Writting the new file
fileout=open(sys.argv[2],'w')
fileout.write(csv_file)
fileout.close()
