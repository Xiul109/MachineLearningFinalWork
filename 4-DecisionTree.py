import sys
import csv
import pydot
import numpy as np
from sklearn import tree
from sklearn.externals.six import StringIO

if(len(sys.argv) is not 3):
	print("Usage: python3 E14.py <input_file> <pdf_file>")
	exit()


data=list(csv.reader(open(sys.argv[1])))
feature_names=data[0][1:]
data=data[1:]

colDamages=feature_names.index('notRepairedDamage')
target=[]
for e in data:
	target.append(data.pop(colDamages))

#Decision Tree Training
treeClas=tree.DecisionTreeClassifier()
treeClas.fit(data, target)

#Decision Tree Plot
dot_data = StringIO()

tree.export_graphviz(treeClas, out_file=dot_data,feature_names=feature_names,class_names=rangesStr,filled=True, rounded=True, special_characters=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf(path=sys.argv[3])
