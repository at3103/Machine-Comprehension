import scipy
import pandas as pd
import numpy as np
import sklearn
import sys
import os
from os import listdir
from os.path import isfile, join
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GroupKFold
from collections import Counter
import traceback

n_x = 21	#Columns which are considered features
n_y = 21 # the column for label

# Load dataset
frames=[]
data_file_path = '../data/featuredata_wo_vectors_classifier/'
data_files = [f for f in listdir(data_file_path) if isfile(join(data_file_path, f)) and f.endswith('.csv')]
for count,i in enumerate(data_files):
	url = data_file_path + i
	dataset = pd.read_csv(url)
	frames.append(dataset)
	final_dataset = pd.concat(frames)
	if count > 100:
		break

#Different buil-in functions of ndarray
print final_dataset.shape, final_dataset.ndim, len(final_dataset.shape) #, dataset.size#, dataset.dtype, dataset.itemsize

#Extracting the values from the dataframe
array = final_dataset.values

#Separating the features and the labels
array = final_dataset.values

#Separating the features and the labels
X = array[:,1:]
Y = array[:,n_y]
qid = array[:,23]


#Enable for labels

class_weight ={}

class_weight['Y'] = 125
class_weight['M'] = 15


print "In whole data",Counter(Y)

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

qid=X_test1[:,23]
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

print "In train",Counter(Y_train)
print "In test",Counter(Y_test)

cv_seed = 7
n_fold = 10
num_of_instances = len(X_train)
scoring = 'accuracy'
scoring1 = 'f1_macro'


#Load the models
models = []
try :
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	#models.append(('SVM', LinearSVC(dual=False,class_weight = class_weight)))
	#models.append(('dt', DecisionTreeClassifier(max_depth=4)))
	models.append(('rf',RandomForestClassifier(n_estimators=20)))
	#models.append(('KNN', KNeighborsClassifier()))


except Exception, e:
	traceback.print_exc()

#Evaluate the models
results = []
names = []
cv_predict=[]
models_eval=[]

for name, model in models:
	kfold = model_selection.KFold(n_splits=2, random_state=cv_seed)
	cv_results = model_selection.cross_val_score(model, X_train[:,:-4], Y_train, cv=2, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	models_eval.append(cv_results.mean())
	print name + " : ", cv_results.mean(),"(",cv_results.std(),")"

# k = 0
# for i in range(0,len(models)):
# 	clf = models[i][1]
# 	clf.fit(X_train[:,k:-4], Y_train)
# 	pred = clf.predict(X_test[:,k:-4])
# 	ac_score = accuracy_score(Y_test, pred)
# 	print models[i][0],"Accuracy is ", ac_score
# 	print "In predicted data",Counter(pred)
# 	features = ['POS','NER','lemmas','dep_tree','F1','span_words', 'q_words', 'ground_truth','predicted_F1_score']
# 	combined_feature = []
# 	for j,item in enumerate(X_test[:,-8:]):
# 		combined_feature.append(list(item))
# 		combined_feature[j].append((pred[j]))

# 	df = pd.DataFrame.from_records(combined_feature, columns = features)
# 	output_file_path = "../data/predictions/classification/latest/"
# 	if not os.path.exists(output_file_path):
# 		os.makedirs(output_file_path)
# 	df.to_csv(os.path.join(output_file_path,models[i][0]+ str(k) +".csv"))

#Hardcoded values
#models_eval = [0.818861651765, 0.794685053429, 0.813897439689]

models_d=[('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('rf',RandomForestClassifier(n_estimators=50))]#, ('KNN', KNeighborsClassifier())]
eclf = VotingClassifier(estimators=models_d, voting='soft', weights=models_eval)
eclf = eclf.fit(X_train[:,:-4],Y_train)
eclf_predict = eclf.predict(X_test[:,:-4])
eclf_predict1 = eclf.predict(X[:,:-4])
eclf_predict2 = eclf.predict(X_train[:,:-4])
print "Accuracy for ecl test:",accuracy_score(Y_test,eclf_predict)
print "Accuracy for ecl whole:",accuracy_score(Y,eclf_predict1)
print "Accuracy for ecl train:",accuracy_score(Y_train,eclf_predict2)

features = ['POS','NER','lemmas','dep_tree','F1','span_words', 'q_words', 'ground_truth','predicted_F1_score']
combined_feature = []
for j,item in enumerate(X_test[:,-8:]):
	combined_feature.append(list(item))
	combined_feature[j].append((eclf_predict[j]))

df = pd.DataFrame.from_records(combined_feature, columns = features)
output_file_path = "../data/predictions/classification/ensemble/"
if not os.path.exists(output_file_path):
	os.makedirs(output_file_path)
df.to_csv(os.path.join(output_file_path,"ensemble_method.csv"))