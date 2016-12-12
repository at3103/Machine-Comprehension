import numpy
import scipy
import pandas
import matplotlib	
import sklearn
import sys
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from util_sush import *
from sklearn.neural_network import MLPClassifier
from kmeans import *
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import resample
#import tensorflow



X = import_train_inp()
Y = import_train_out()


#Oversample
#X,Y = oversample(X, Y)
X,Y = resample(X, Y, n_samples = 20000)

#Validation Size
test_size = 0.3

#Set the seed for randomness here
seed = 7

#Obtain the training and test sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,
	test_size = test_size, random_state = seed)

#Evaluation metrics and test options

cv_seed = 7
n_fold = 10
num_of_instances = len(X_train)
scoring = 'accuracy'
scoring1 = 'f1_macro'


#Load the models
models = []

models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('rf',RandomForestClassifier(n_estimators=1000)))


#Models_Evaluation
models_eval=[]
models_metrics=[]

#Evaluate the models
results = []
names = []
cv_predict=[]
# for name, model in models:
# 	if name == 'rf':
# 		n_fold = 2
# 	kfold = model_selection.KFold(n_splits=n_fold, random_state=cv_seed)
# 	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=n_fold, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	models_eval.append(cv_results.mean())
# 	print "Done with CV of " + name + " with mean ", cv_results.mean()
# 	predictions=[]
# 	predictions1=[]

models_eval = [0.451013033819, 0.398122086505, 0.479197402009]
models_eval1 = [0.451013033819, 0.479197402009]
# adb = AdaBoostClassifier(n_estimators=300)
# adb.fit(X_train,Y_train)
# adb_predict = adb.predict(X_test)
# adb_predict1 = adb.predict(X)
# adb_predict2 = adb.predict(X_train)
# adb.fit(X,Y)
# adb_predict3 = adb.predict(X)
# print "Accuracy for Adaboost test:",accuracy_score(Y_test,adb_predict)
# print "Accuracy for Adaboost whole:",accuracy_score(Y,adb_predict1)
# print "Accuracy for Adaboost train:",accuracy_score(Y_train,adb_predict2)
# print "Accuracy for Adaboost whole1:",accuracy_score(Y,adb_predict3)

url = "public_test.csv"
dataset = pandas.read_csv(url)
array = dataset.values



models_d=[('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('rf',RandomForestClassifier(n_estimators=1000))]
models_d1=[('LR', LogisticRegression()), ('rf',RandomForestClassifier(n_estimators=1000))]
eclf = VotingClassifier(estimators=models_d, voting='soft', weights=models_eval)
eclf = eclf.fit(X_train,Y_train)
eclf_predict = eclf.predict(X_test)
eclf_predict1 = eclf.predict(X)
eclf_predict2 = eclf.predict(X_train)
eclf.fit(X,Y)
p = eclf.predict(X)
print "Accuracy for ecl test:",accuracy_score(Y_test,eclf_predict)
print "Accuracy for ecl whole:",accuracy_score(Y,eclf_predict1)
print "Accuracy for ecl train:",accuracy_score(Y_train,eclf_predict2)
print "Accuracy for ecl whole1:",accuracy_score(Y,p)


pr2 = eclf.predict(array)
#t = numpy.concatenate((array[:,0],pr2),axis=1)
df = pandas.DataFrame.from_records(t)
df.to_csv('Output_public.csv')

adb_pr2 = adb.predict(array)
#t_adb = numpy.concatenate((array[:,0],adb_pr2),axis=1)
df_adb = pandas.DataFrame.from_records(adb_pr2)
df_adb.to_csv('Output_public_adb.csv')

'''
Reference :	1. http://machinelearningmastery.com/machine-learning-in-python-step-by-step/
			2. http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
'''