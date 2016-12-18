import numpy
import scipy
import matplotlib	
import sklearn
import sys
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.model_selection import GroupKFold
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
#from sklearn.ensemble.forest import ExtraTreeRegressor, ExtraTreesRegressor
#from sklearn.linear_model.huber import HuberRegressor
#from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model.base import LinearRegression
#from sklearn.svm.classes import LinearSVR, NuSVR
#from sklearn.linear_model.ransac import RANSACRegressor
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model.ridge import Ridge, RidgeCV
#from sklearn.linear_model.stochastic_gradient import SGDRegressor

	
n_x = 22	#Columns which are considered features
n_y = 22 # the column for label

# Load dataset
frames = []

data_file_path = "../data/featuredata_br/"
data_files = [f for f in listdir(data_file_path) if isfile(join(data_file_path, f)) and f.endswith('.csv')]
for i in data_files:
	url = data_file_path + i
	dataset = pd.read_csv(url)
	frames.append(dataset)
	final_dataset = pd.concat(frames)

#Different buil-in functions of ndarray
print final_dataset.shape, final_dataset.ndim, len(final_dataset.shape)

#Extracting the values from the dataframe
array = final_dataset.values

#Separating the features and the labels
X = array[:,1:]
Y = array[:,n_y]
qid = array[:,24]


#Set the seed for randomness here
gkf = GroupKFold(n_splits=2)

#Obtain the training and test sets
X_train = []
X_test = []
Y_train = []
Y_test = []

splits = gkf.split(X,Y,groups = qid)
cur_splits = splits.next()
X_train = X[cur_splits[0]]
X_test1 = X[cur_splits[1]]
Y_train = Y[cur_splits[0]]
Y_test1 = Y[cur_splits[1]]

qid=X_test1[:,24]
splits = gkf.split(X_test1,Y_test1,groups = qid)
cur_splits = splits.next()

print X_train.shape, X_test1[cur_splits[0]].shape
print Y_train.shape, Y_test1[cur_splits[0]].shape

X_train = (np.concatenate((X_train.T, X_test1[cur_splits[0]].T),axis = 1)).T
X_test = X_test1[cur_splits[1]]
Y_train = (np.concatenate((Y_train,Y_test1[cur_splits[0]])))
Y_test = Y_test1[cur_splits[1]]

print X_train.shape, X_test.shape
print Y_train.shape, Y_test.shape

name_list =['ExtraTreeRegressor', 'ExtraTreesRegressor', 
'GradientBoostingRegressor', 'HuberRegressor', 'KernelRidge', 'LinearRegression', 'LinearSVR', 
'NuSVR', 'RANSACRegressor', 'RandomForestRegressor', 'Ridge', 'RidgeCV', 'SGDRegressor', 'SVR']

#Load the models
models = []
# models.append(('ExtraTreeRegressor', ExtraTreeRegressor()))
# models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
# #models.append(('GradientBoostingRegressor', GradientBoostingRegressor()))
# models.append(('HuberRegressor', HuberRegressor()))
# models.append(('KernelRidge', KernelRidge()))
# models.append(('LinearRegression', LinearRegression()))
# models.append(('LinearSVR', LinearSVR()))
# models.append(('NuSVR', NuSVR()))
# models.append(('RANSACRegressor', RANSACRegressor()))
# models.append(('RandomForestRegressor', RandomForestRegressor()))
# models.append(('Ridge', Ridge()))
# models.append(('RidgeCV', RidgeCV()))
# models.append(('SGDRegressor', SGDRegressor()))


pred = []
name = 'Ridge'
#for name,model in models:
for n in [10,50,200,400,600]:
	reg = AdaBoostRegressor(n_estimators=n, base_estimator=Ridge())

	reg.fit(X_train[:,:-4], Y_train) 
	pred = reg.predict(X_test[:,:-4])
	ac_score = mean_squared_error(Y_test, pred)
	print "AdaboostRegressor for "+name+" MSE is ", ac_score, "and n = ",n
	features = ['span_words', 'q_words', 'ground_truth','predicted_F1_score']
	combined_feature = []
	for j,item in enumerate(X_test[:,-3:]):
		combined_feature.append(list(item))
		combined_feature[j].append((pred[j]))
	df = pd.DataFrame.from_records(combined_feature, columns = features)
	output_file_path = "../data/predictions/Adabosot/"
	if not os.path.exists(output_file_path):
		os.makedirs(output_file_path)
	df.to_csv(os.path.join(output_file_path,'AdaboostRegressor_prediction_'+name+str(n)+'.csv'))
