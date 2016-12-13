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

#To check the version of python
#print "Python : {}".format(sys.version)
	
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

#Evaluation metrics and test options

cv_seed = 7
n_fold = 10
num_of_instances = len(X_train)
scoring = 'accuracy'
scoring1 = 'f1_macro'


#Load the models
models = []

i =1
alph = [1e-01,1e-02,1e-03,1e-04,1e-05,1e-07,1,0.5,0.000004]
lrn = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.002,0.4,0.3,0.0000006]
for a in alph:
	for l in lrn:

	# clf = SGDRegressor(alpha=alph, average=False, epsilon=0.1, eta0=0.01,
 #         fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
 #         loss='squared_loss', n_iter=5, penalty='l2', power_t=0.25,
 #         random_state=None, shuffle=True, verbose=0, warm_start=False)

		clf = MLPRegressor(activation='relu', alpha=1e-05, batch_size=500,
		       beta_1=0.9, beta_2=0.1, early_stopping=False,
		       epsilon=a, hidden_layer_sizes=(12, 6), learning_rate='adaptive',
		       learning_rate_init=l, max_iter=2000, momentum=0.9,
		       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
		       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
		       warm_start=False)

		#Y_train.reshape(Y_train.ndim,1)
		clf.fit(X_train[:,:-4], Y_train)
		pred = clf.predict(X_test[:,:-4])
		#print "Accuracy of clf BoM is ", accuracy_score(Y_test, pred_nn_bom)
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
		df.to_csv(os.path.join(output_file_path,'MLP_prediction_alph'+str(a)+"_"+ str(l)'.csv'))
		i += 1
