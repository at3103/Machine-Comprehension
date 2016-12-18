import scipy
import pandas as pd
import matplotlib	
import sklearn
import sys
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVR
from kmeans import *
from sklearn import linear_model
import numpy as np
import os
from os import listdir
from os.path import isfile, join

n_x = 22	#Columns which are considered features
n_y = 22 # the column for label

# Load dataset
frames=[]
data_file_path = "../data/featuredata_br_copy/"
data_files = [f for f in listdir(data_file_path) if isfile(join(data_file_path, f)) and f.endswith('.csv')]
for i in data_files:
	url = data_file_path + i
	dataset = pd.read_csv(url)#, names=names)
	frames.append(dataset)
	final_dataset = pd.concat(frames)

#Different buil-in functions of ndarray
print final_dataset.shape, final_dataset.ndim, len(final_dataset.shape) #, dataset.size#, dataset.dtype, dataset.itemsize

#Extracting the values from the dataframe
array = final_dataset.values

#Separating the features and the labels
array = final_dataset.values
Y = array[:,n_y]

for i in range(len(Y)):
	if float(Y[i]) == 1.0:
		Y[i] = 'A'
	elif float(Y[i]) >= 0.94: 
		Y[i] = 'B'
	elif float(Y[i]) >= 0.90:
		Y[i] = 'C'
	elif float(Y[i]) >= 0.87:
		Y[i] = 'D'
	elif float(Y[i]) >= 0.84:
		Y[i] = 'E'
	elif float(Y[i]) >= 0.80:
		Y[i] = 'F'
	elif float(Y[i]) >= 0.75:
		Y[i] = 'G'
	elif float(Y[i]) >= 0.70:
		Y[i] = 'H'		
	elif float(Y[i]) >= 0.65:
		Y[i] = 'I'		
	elif float(Y[i]) >= 0.60:
		Y[i] = 'J'		
	elif float(Y[i]) >= 0.50:
		Y[i] = 'K'		
	elif float(Y[i]) >= 0.40:
		Y[i] = 'L'		
	elif float(Y[i]) >= 0.30:
		Y[i] = 'M'
	elif float(Y[i]) >= 0.20:
		Y[i] = 'N'
	elif float(Y[i]) >= 0.10:
		Y[i] = 'O'			
	elif float(Y[i]) >= 0.08:
		Y[i] = 'P'
	elif float(Y[i]) >= 0.05:
		Y[i] = 'Q'
	elif float(Y[i]) == 0.0:
		Y[i] = 'S'	
	else:
		Y[i] = 'R'

#Separating the features and the labels
X = array[:,1:]

df = pd.DataFrame.from_records(X)
output_file_path = "../data/predictions/lr/check/"
if not os.path.exists(output_file_path):
	os.makedirs(output_file_path)
df.to_csv(os.path.join(output_file_path,"Full_data_classification.csv"))

df1 = pd.DataFrame.from_records(Y)
df1.to_csv(os.path.join(output_file_path,"Full_data_classification_Y.csv"))


