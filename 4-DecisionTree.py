import sys
import csv
import pydotplus
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

if(len(sys.argv) is not 3):
	print("Usage: python3 E14.py <input_file> <pdf_file>")
	exit()

data=pd.read_csv(sys.argv[1], sep=',',encoding='Latin1')

#Encoder
campos=['vehicleType','model','brand','gearbox','fuelType','notRepairedDamage']
encoders=[]
for c in campos:
	le = preprocessing.LabelEncoder()
	le.fit(data[c])
	data[c]=le.transform(data[c])
	encoders.append(le)


#Decision Tree Training
notRD=data['notRepairedDamage']
data=data.drop('notRepairedDamage',1)
treeClas=tree.DecisionTreeClassifier(min_impurity_split=0.12,max_depth=7)
treeClas.fit(data, notRD)

#Decision Tree Plot
dot_data = StringIO()

tree.export_graphviz(treeClas, out_file=dot_data,feature_names=list(data),class_names=['ja', 'nein'],filled=True, rounded=True, special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf(path=sys.argv[2])
