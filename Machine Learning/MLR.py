import numpy
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
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from kmeans import *
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
import pandas
import numpy as np
import os

n_x = 22	#Columns which are considered features
n_y = 22 # the column for label

# Load dataset
frames = []

data_file_path = "../data/featuredata_br/"
data_files = [f for f in listdir(data_file_path) if isfile(join(data_file_path, f)) and f.endswith('.csv')]
for i in data_files:
	url = data_file_path + i
	# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = pandas.read_csv(url)#, names=names)
	frames.append(dataset)
	final_dataset = pd.concat(frames)

#Different buil-in functions of ndarray
print final_dataset.shape, final_dataset.ndim, len(final_dataset.shape) #, dataset.size#, dataset.dtype, dataset.itemsize

#Extracting the values from the dataframe
array = final_dataset.values

#Separating the features and the labels
X = array[:,1:]
Y = array[:,n_y]
qid = array[:,24]
#print X[0], Y[0], qid[0]
#Validation Size
test_size = 0.3

#Set the seed for randomness here
seed = 7
gkf = GroupKFold(n_splits=2)
#Obtain the training and test sets
X_train = []
X_test = []
Y_train = []
Y_test = []
#X_train, X_test, Y_train, Y_test = gkf.split(X,Y,groups = qid)
splits = gkf.split(X,Y,groups = qid)
cur_splits = splits.next()
X_train = X[cur_splits[0]]
X_test = X[cur_splits[1]]
Y_train = Y[cur_splits[0]]
Y_test = Y[cur_splits[1]]

#Enable for labels

# for i in range(len(Y_train)):
# 	if float(Y_train[i]) == 1.0:
# 		Y_train[i] = 'Y'
# 	elif float(Y_train[i]) >= 0.25:
# 		Y_train[i] = 'M'
# 	else:
# 		Y_train[i] = 'N'

# for i in range(len(Y_test)):
# 	if float(Y_test[i]) == 1.0:
# 		Y_test[i] = 'Y'
# 	elif float(Y_test[i]) >= 0.25:
# 		Y_test[i] = 'M'
# 	else:
# 		Y_test[i] = 'N'



#Y_train = list(Y_train)

#Y = np.asarray(array[n_y], dtype="|S6")



'''
#Preparing the metrics for evaluation and cross-validation
'''

i =1
alph = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
lrn = [ 0.01, 0.02, 0.03, 0.00001, 0.0001]
for a in alph:
	for l in lrn:
		clf = MLPRegressor(activation='relu', alpha=1e-05, batch_size=500,
		       beta_1=0.9, beta_2=0.1, early_stopping=False,
		       epsilon=a, hidden_layer_sizes=(12, 6), learning_rate='adaptive',
		       learning_rate_init=l, max_iter=2000, momentum=0.9,
		       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
		       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
		       warm_start=False)

		clf.fit(X_train[:,:-4], Y_train)
		pred = clf.predict(X_test[:,:-4])
		combined_feature = []
		for j,item in enumerate(X_test):
			combined_feature.append(list(item))
			combined_feature[j].append((pred[j]))
		ac_score = mean_squared_error(Y_test, pred)

		print "MLPRegressor MSE is ", ac_score, "Epsilon: ",a, " Learning rate: ", l

		features = ['root match 1', 'sent_root_qs', 'qs_root_sent',  'n_wrds_l', 'n_wrds_r', 
					'n_wrds_in', 'n_wrds_sent', 'm_u_sent', 'm_u_span', 'm_u_l', 'm_u_r', 'span_wf', 
					'm_b_sent', 'm_b_span', 'm_b_l', 'm_b_r', 'constituent_label', 'pos', 'ner', 'lemma', 'deptree_path', 'F1_score',
					'span_words', 'q_words', 'ground_truth','predicted_F1_score']


		df = pandas.DataFrame.from_records(combined_feature, columns = features)
		output_file_path = "../data/predictions/mlcp/new"
		if not os.path.exists(output_file_path):
			os.makedirs(output_file_path)
		df.to_csv(os.path.join(output_file_path,'MLP_prediction_alph'+str(a)+"_"+ str(l) + '.csv'))
		i += 1