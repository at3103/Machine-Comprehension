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
data_file_path = "../data/featuredata_br/"
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

#Separating the features and the labels
X = array[:,1:]
Y = array[:,n_y]
qid = array[:,24]

k=1

df = pd.DataFrame.from_records(X)
output_file_path = "../data/predictions/lr/check/"
if not os.path.exists(output_file_path):
	os.makedirs(output_file_path)
df.to_csv(os.path.join(output_file_path,"Full_data_check"+ str(k)+".csv"))

