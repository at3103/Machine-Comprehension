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

from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from kmeans import *
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
import pandas
import numpy as np
import os
from os import listdir
from os.path import isfile, join


#To check the version of python
#print "Python : {}".format(sys.version)
	
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

#To take a quick peek
#print dataset.head(20)

#To print summary
#print dataset.describe()

#To know the different classes of the dataset
#print dataset.groupby('Species').size()
#print dataset.groupby('Sepal.Width').size()

#Data Visualization:  Plots

#Box Plot
#dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
#plt.show()

#Histogram
#dataset.hist()
#plt.show()


#Scatter Plot
#scatter_matrix(dataset)
#plt.show()
'''

'''
#Extracting test and train data sets from given data set




#Extracting the values from the dataframe
array = final_dataset.values
#print array[0]
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
#X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_qid_uniq,
#	test_size = test_size, random_state = seed)
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

print X_train.shape, X_test.shape
print Y_train.shape, Y_test.shape

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

#print Y_train.dtype

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

# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('SVM', LinearSVC(dual=False,class_weight = 'balanced')))
#models.append(('dt', DecisionTreeClassifier(max_depth=4)))
#models.append(('rf',RandomForestClassifier(n_estimators=1000)))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('KNN', KNeighborsClassifier()))
# models.append(('SVM', SVC(kernel='rbf', probability=True)))
#models.append(('SVR', SVR(kernel='linear', C=1e3)))
models.append(('LinearRegression', linear_model.LinearRegression()))
models.append(('BayesianRegression', linear_model.BayesianRidge()))
#models.append(('Perceptron', linear_model.Perceptron()))
#Models_Evaluation
models_eval=[]
models_metrics=[]

#Evaluate the models
results = []
names = []
cv_predict=[]
# for i,value in enumerate(Y_train):
#  	if Y_train[i] < 1.0:
#  		Y_train[i] = 0.0
# Y_train = np.asarray(Y_train, dtype="f4")
#print Y_train
#print len(Y_train),len(X_train)
for name, model in models:
	names.append(name)

'''
for i in range(0,len(names)):
	models[i][1].fit(X_train, Y_train) 
	predictions.append(models[i][1].predict(X_test))
	predictions1.append(models[i][1].predict(X))
	ac_score = accuracy_score(Y_test, predictions[i])
	ac_score1 = accuracy_score(Y,predictions1[i])
	conf_matrix = confusion_matrix(Y_test, predictions[i])
	class_rep = classification_report(Y_test, predictions[i])	
	models_metrics.append([names[i],ac_score, conf_matrix, class_rep])
	print names[i],"Accuracy is ", ac_score
	print names[i], "Accuracy for whole dataset is ", ac_score1
'''
# for i,value in enumerate(Y_test):
#  	if Y_test[i] < 1.0:
#  		Y_test[i] = 0.0
# Y_test = np.asarray(Y_test, dtype="f4")

pred = []
for k in range(0,n_x):
	for i in range(0,len(names)):
		models[i][1].fit(X_train[:,k:-4], Y_train) 
		pred = models[i][1].predict(X_test[:,k:-4])
		print "Prediction"
		print pred
		ac_score = mean_squared_error(Y_test, pred)
		print names[i],"Accuracy is ", ac_score

		features = ['root match 1', 'sent_root_qs', 'qs_root_sent',  'n_wrds_l', 'n_wrds_r', 
					'n_wrds_in', 'n_wrds_sent', 'm_u_sent', 'm_u_span', 'm_u_l', 'm_u_r', 'span_wf', 
					'm_b_sent', 'm_b_span', 'm_b_l', 'm_b_r', 'constituent_label', 'pos', 'ner', 'lemma', 'deptree_path', 'F1_score',
					'span_words', 'q_words', 'ground_truth','predicted_F1_score']
		combined_feature = []
		for j,item in enumerate(X_test):
			combined_feature.append(list(item))
			combined_feature[j].append((pred[j]))
		df = pandas.DataFrame.from_records(combined_feature, columns = features)
		output_file_path = "../data/predictions/lr_new_total_data/"
		if not os.path.exists(output_file_path):
			os.makedirs(output_file_path)
		df.to_csv(os.path.join(output_file_path,names[i]+'_predictionb_'+str(k)+'.csv'))

'''

pred_BOM = predictions[0]
pred1_BOM = predictions1[0]
'''
# for i in models_metrics:
# 	print i[0], i[1]
# 	print i[2]
# 	print "hey",i[3]

'''pred_Array = numpy.zeros(8)
for i in range(0,len(pred_BOM)):
	pred_Array = numpy.zeros(8)
	for j in range(0,len(predictions)):
		pred_Array[predictions[j][i]] += results[j].mean()
	pred_BOM[i] = numpy.argmax(pred_Array)

for i in range(0,len(pred1_BOM)):
	pred1_Array = numpy.zeros(8)
	for j in range(0,len(predictions1)):
		pred1_Array[predictions1[j][i]] += results[j].mean()
	pred1_BOM[i] = numpy.argmax(pred1_Array)
print "Accuracy of BoM is ", accuracy_score(Y_test,pred_BOM)
print "Accuracy of BoM for whole data_setis ", accuracy_score(Y,pred1_BOM)'''
#('SVM', SVC(kernel='rbf', probability=True, class_weight = 'balanced')), ('dt', DecisionTreeClassifier(max_depth=4)),


'''
models_d=[('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('rf',RandomForestClassifier(n_estimators=1000))]#, ('KNN', KNeighborsClassifier())]
eclf = VotingClassifier(estimators=models_d, voting='soft', weights=models_eval)
eclf = eclf.fit(X_train,Y_train)
eclf_predict = eclf.predict(X_test)
eclf_predict1 = eclf.predict(X)
eclf_predict2 = eclf.predict(X_train)
print "Accuracy for ecl test:",accuracy_score(Y_test,eclf_predict)
print "Accuracy for ecl whole:",accuracy_score(Y,eclf_predict1)
print "Accuracy for ecl train:",accuracy_score(Y_train,eclf_predict2)
'''
# i = 1
# alph = [1e-01,1e-02,1e-03,1e-04,1e-05,1e-07,1,0.5,0.000004]
# lrn = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.002,0.4,0.3,0.0000006]
# for a in alph:
# 	for l in lrn:

# 		print "i is ",i," Learning rate: ",l," Alpha is ", a
	# clf = SGDRegressor(alpha=alph, average=False, epsilon=0.1, eta0=0.01,
 #         fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
 #         loss='squared_loss', n_iter=5, penalty='l2', power_t=0.25,
 #         random_state=None, shuffle=True, verbose=0, warm_start=False)

		# clf = MLPRegressor(activation='relu', alpha=1e-05, batch_size=500,
		#        beta_1=0.9, beta_2=0.1, early_stopping=False,
		#        epsilon=a, hidden_layer_sizes=(12, 6), learning_rate='adaptive',
		#        learning_rate_init=l, max_iter=2000, momentum=0.9,
		#        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
		#        solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
		#        warm_start=False)

		# #Y_train.reshape(Y_train.ndim,1)
		# clf.fit(X_train[:,:-4], Y_train)
		# pred = clf.predict(X_test[:,:-4])
		# #print "Accuracy of clf BoM is ", accuracy_score(Y_test, pred_nn_bom)
		# combined_feature = []
		# for j,item in enumerate(X_test):
		# 	#print item
		# 	combined_feature.append(list(item))
		# 	#combined_feature[j].append(float(pred[j]))
		# 	combined_feature[j].append((pred[j]))
		# ac_score = mean_squared_error(Y_test, pred)

		# print "MLPRegressor Accuracy is ", ac_score

		# features = ['root match 1', 'sent_root_qs', 'qs_root_sent',  'n_wrds_l', 'n_wrds_r', 
		# 			'n_wrds_in', 'n_wrds_sent', 'm_u_sent', 'm_u_span', 'm_u_l', 'm_u_r', 'span_wf', 
		# 			'm_b_sent', 'm_b_span', 'm_b_l', 'm_b_r', 'constituent_label', 'pos', 'ner', 'lemma', 'deptree_path', 'F1_score',
		# 			'span_words', 'q_words', 'ground_truth','predicted_F1_score']


		# df = pandas.DataFrame.from_records(combined_feature, columns = features)
		# output_file_path = "../data/predictions/mlcp/new"
		# if not os.path.exists(output_file_path):
		# 	os.makedirs(output_file_path)
		# df.to_csv(os.path.join(output_file_path,'MLP_prediction_alph'+str(i)+'.csv'))
		# i += 1


# i = 1
# lrn = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.002]
# for l in lrn:
# 	clf = GradientBoostingRegressor(n_estimators=100, learning_rate=l,
# 	max_depth=1, random_state=0, loss='ls')
# 	#Y_train.reshape(Y_train.ndim,1)
# 	clf.fit(X_train[:,:-4], Y_train)
# 	pred = clf.predict(X_test[:,:-4])
# 	#print "Accuracy of clf BoM is ", accuracy_score(Y_test, pred_nn_bom)
# 	combined_feature = []
# 	for j,item in enumerate(X_test):
# 		#print item
# 		combined_feature.append(list(item))
# 		#combined_feature[j].append(float(pred[j]))
# 		combined_feature[j].append((pred[j]))
# 	ac_score = mean_squared_error(Y_test, pred)

# 	print "GBRegressor Accuracy is ", ac_score

# 	features = ['root match 1', 'sent_root_qs', 'qs_root_sent',  'n_wrds_l', 'n_wrds_r', 
# 				'n_wrds_in', 'n_wrds_sent', 'm_u_sent', 'm_u_span', 'm_u_l', 'm_u_r', 'span_wf', 
# 				'm_b_sent', 'm_b_span', 'm_b_l', 'm_b_r', 'constituent_label', 'pos', 'ner', 'lemma', 'deptree_path', 'F1_score',
# 				'span_words', 'q_words', 'ground_truth','predicted_F1_score']


# 	df = pandas.DataFrame.from_records(combined_feature, columns = features)
# 	output_file_path = "../data/predictions/gbr/"
# 	if not os.path.exists(output_file_path):
# 		os.makedirs(output_file_path)
# 	df.to_csv(os.path.join(output_file_path,'GBR_prediction_alph'+str(i)+'.csv'))
# 	i += 1



# clf = LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', 
# fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, 
# random_state=None, max_iter=1000)
# #Y_train.reshape(Y_train.ndim,1)
# clf.fit(X_train[:,:-4], Y_train)
# pred = clf.predict(X_test[:,:-4])
# #print "Accuracy of clf BoM is ", accuracy_score(Y_test, pred_nn_bom)
# combined_feature = []
# for j,item in enumerate(X_test):
# 	#print item
# 	combined_feature.append(list(item))
# 	#combined_feature[j].append(float(pred[j]))
# 	combined_feature[j].append((pred[j]))
# ac_score = mean_squared_error(Y_test, pred)

# print "LSVRegressor Accuracy is ", ac_score

# features = ['root match 1', 'sent_root_qs', 'qs_root_sent',  'n_wrds_l', 'n_wrds_r', 
# 			'n_wrds_in', 'n_wrds_sent', 'm_u_sent', 'm_u_span', 'm_u_l', 'm_u_r', 'span_wf', 
# 			'm_b_sent', 'm_b_span', 'm_b_l', 'm_b_r', 'constituent_label', 'pos', 'ner', 'lemma', 'deptree_path', 'F1_score',
# 			'span_words', 'q_words', 'ground_truth','predicted_F1_score']


# df = pandas.DataFrame.from_records(combined_feature, columns = features)
# output_file_path = "../data/predictions/lsvr/"
# if not os.path.exists(output_file_path):
# 	os.makedirs(output_file_path)
# df.to_csv(os.path.join(output_file_path,'LSVR_prediction_alph'+str(i)+'.csv'))



#a = numpy.array([(1,2,3),(2,3,1)])
#Different buil-in functions of ndarray
#print a.ndim, a.shape, len(a.shape), a.size, a.dtype, a.itemsize
#print a.data
#b = numpy.arange(20).reshape(10,2)




'''
Reference :	1. http://machinelearningmastery.com/machine-learning-in-python-step-by-step/
			2. http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
'''