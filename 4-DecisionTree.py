#imports
import sys
import csv
import pydotplus
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, y_true, y_pred, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(set(y_true)))
	plt.xticks(tick_marks, set(y_pred), rotation=45)
	plt.yticks(tick_marks, set(y_true))
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

#Aruments management
if(len(sys.argv) is not 3):
	print("Usage: python3 E14.py <input_file> <pdf_file>")
	exit()

#File reading
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
max_depth=7
notRD=data['notRepairedDamage']
data=data.drop('notRepairedDamage',1)
treeClas=tree.DecisionTreeClassifier(max_depth=max_depth)
treeClas.fit(data, notRD)

yP=treeClas.predict(data)

#Confusion Matrix compute
cm = confusion_matrix(y_true=notRD, y_pred=yP)

#Confusion matrix plot
print(cm)
plot_confusion_matrix(cm, notRD,yP)
plt.show()

#Decision Tree Plot if it isn't to big
if max_depth==None || max_depth>10
	print("Tree to big to be plotted")
else
	dot_data = StringIO()

	tree.export_graphviz(treeClas, out_file=dot_data,feature_names=list(data),class_names=['ja', 'nein'],filled=True, rounded=True, special_characters=True)

	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf(path=sys.argv[2])
