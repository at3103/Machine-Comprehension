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

#To check the version of python
#print "Python : {}".format(sys.version)
	
n_x = 4	#Columns which are considered features
n_y = 5 # the column for label
# Load dataset
frames = []
for i in range(100):
	url = ".csv"
	# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = pandas.read_csv(url)#, names=names)
	frames.append(dataset)
	final_dataset = pd.concat(frames)




#Different buil-in functions of ndarray
print dataset.shape, dataset.ndim, len(dataset.shape), dataset.size#, dataset.dtype, dataset.itemsize

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
dataset.hist()
plt.show()


#Scatter Plot
scatter_matrix(dataset)
plt.show()
'''

'''
#Extracting test and train data sets from given data set




#Extracting the values from the dataframe
array = dataset.values

#Separating the features and the labels
X = array [:,0:n_x]
Y = array[:,n_y]


#Validation Size
test_size = 0.3

#Set the seed for randomness here
seed = 7

#Obtain the training and test sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,
	test_size = test_size, random_state = seed)


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

models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('SVM', LinearSVC(dual=False,class_weight = 'balanced')))
#models.append(('dt', DecisionTreeClassifier(max_depth=4)))
#models.append(('rf',RandomForestClassifier(n_estimators=1000)))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('SVM', SVC(kernel='rbf', probability=True)))

#Models_Evaluation
models_eval=[]
models_metrics=[]

#Evaluate the models
results = []
names = []
cv_predict=[]
for name, model in models:
	kfold = model_selection.KFold(n_splits=n_fold, random_state=cv_seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=n_fold, scoring=scoring)
	#predicted = cross_val_predict(model, X_train, Y_train, cv=10)
	#cv_predict.append(*predicted)
	#cv_results1= model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	models_eval.append(cv_results.mean())
	#print name + " : ", cv_results.mean(),"(",cv_results.std(),")"
	predictions=[]
	predictions1=[]


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

for i in range(0,len(names)):
	models[i][1].fit(X_train, Y_train) 
	pred = models[i][1].predict(X_test)
	pred1 = models[i][1].predict(X)
	ac_score = accuracy_score(Y_test, pred)
	ac_score1 = accuracy_score(Y,pred1)
	#conf_matrix = confusion_matrix(Y_test, predictions)
	print names[i],"Accuracy is ", ac_score
	print names[i], "Accuracy for whole dataset is ", ac_score1

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

'''clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size=500,
       beta_1=0.9, beta_2=0.1, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(8, 4), learning_rate='constant',
       learning_rate_init=0.00001, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
'''
#clf.fit(cv_predict, Y_train)
#pred_nn_bom = clf.predict(predictions)
#print "Accuracy of clf BoM is ", accuracy_score(Y_test, pred_nn_bom)

#a = numpy.array([(1,2,3),(2,3,1)])
#Different buil-in functions of ndarray
#print a.ndim, a.shape, len(a.shape), a.size, a.dtype, a.itemsize
#print a.data
#b = numpy.arange(20).reshape(10,2)




'''
Reference :	1. http://machinelearningmastery.com/machine-learning-in-python-step-by-step/
			2. http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
'''