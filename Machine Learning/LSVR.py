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
for i in xrange(1,38):
	url = "../data/featuredata/{0}.csv".format(i)
	# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = pandas.read_csv(url)#, names=names)
	frames.append(dataset)
	final_dataset = pd.concat(frames)

#Different buil-in functions of ndarray
print dataset.shape, dataset.ndim, len(dataset.shape) #, dataset.size#, dataset.dtype, dataset.itemsize

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

clf = LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', 
fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, 
random_state=None, max_iter=1000)
clf.fit(X_train[:,:-4], Y_train)
pred = clf.predict(X_test[:,:-4])
combined_feature = []
for j,item in enumerate(X_test):
	combined_feature.append(list(item))
	combined_feature[j].append((pred[j]))
ac_score = mean_squared_error(Y_test, pred)

print "LSVRegressor Accuracy is ", ac_score

features = ['root match 1', 'sent_root_qs', 'qs_root_sent',  'n_wrds_l', 'n_wrds_r', 
			'n_wrds_in', 'n_wrds_sent', 'm_u_sent', 'm_u_span', 'm_u_l', 'm_u_r', 'span_wf', 
			'm_b_sent', 'm_b_span', 'm_b_l', 'm_b_r', 'constituent_label', 'pos', 'ner', 'lemma', 'deptree_path', 'F1_score',
			'span_words', 'q_words', 'ground_truth','predicted_F1_score']


df = pandas.DataFrame.from_records(combined_feature, columns = features)
output_file_path = "../data/predictions/lsvr/"
if not os.path.exists(output_file_path):
	os.makedirs(output_file_path)
df.to_csv(os.path.join(output_file_path,'LSVR_prediction_LVSR.csv'))

