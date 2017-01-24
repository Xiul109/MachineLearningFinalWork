import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn import metrics

all_data = pd.read_csv('super_clean.csv', sep=',',encoding='Latin1')

print (all_data['price'].size)

numpyData=np.array([all_data['price'],all_data['kilometer'],all_data['antiguedad']]).T

min_max_scaler = preprocessing.MinMaxScaler()
numpyData = min_max_scaler.fit_transform(numpyData)

db=DBSCAN(eps=0.015,min_samples=8,metric='euclidean')
ydb=db.fit_predict(numpyData)
print (max(ydb))
