#Import
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys

if(len(sys.argv) is not 4):
	print("Usage:   python3 1-DataCleaning.py <input_file> <output_file> <%file reduction [0-1]>")
	print("Example: python3 1-DataCleaning.py autos.csv output_file.csv 0.5")
	exit()

all_data = pd.read_csv(sys.argv[1], sep=',',encoding='Latin1')

#Preprocessing
work_data=all_data.drop('nrOfPictures',1)
work_data=work_data.drop('abtest',1)
work_data=work_data.drop('dateCreated',1)

work_data=work_data.drop('postalCode',1)
work_data=work_data.drop('dateCrawled',1)
work_data=work_data.drop('name',1)

work_data = work_data[work_data.seller != 'gewerblich']
work_data=work_data.drop('seller',1)

work_data = work_data[work_data.offerType != 'Gesuch']
work_data=work_data.drop('offerType',1)

#Creating field antiguedad
aux=pd.DataFrame()
aux['months']=work_data['lastSeen'].str.split('-',expand=True)[1].astype(int)-work_data['monthOfRegistration']
aux['years']=work_data['lastSeen'].str.split('-',expand=True)[0].astype(int)-work_data['yearOfRegistration']

aux['years']=aux.apply(lambda x: x['years']-1 if x['months']<0 else x['years'],axis=1)

work_data['antiguedad']=aux.apply(lambda x: (x['years']*12)+(x['months']%12),axis=1)
work_data=work_data[work_data.antiguedad >0]

del aux, all_data

#Data cleaning
work_data=work_data.drop_duplicates()

work_data = work_data[work_data.fuelType != 'andere']
work_data = work_data[work_data.model != 'andere']

work_data = work_data[(work_data.price >= 200)&(work_data.price<100000)]

work_data = work_data[(work_data.powerPS > 35) & (work_data.powerPS < 1000)]

work_data=work_data[work_data.kilometer < 500000]

work_data = work_data[(work_data.yearOfRegistration >= 1980) & (work_data.yearOfRegistration < 2017)]

work_data=work_data.drop('lastSeen',1)
work_data=work_data.drop('monthOfRegistration',1)
work_data=work_data.drop('yearOfRegistration',1)

work_data=work_data.drop_duplicates()

cleaned_data = work_data

superclean_data = cleaned_data.dropna()

#Data dimension reduction
superclean_data = superclean_data.sample(frac=float(sys.argv[3]))

#Writing file
csv_file_superClean=superclean_data.to_csv(index=False,float_format="%f")

fileout=open(sys.argv[2],'w')
fileout.write(csv_file_superClean)
