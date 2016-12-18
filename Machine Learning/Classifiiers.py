import scipy
import pandas as pd
import numpy as np
import matplotlib	
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

n_x = 22	#Columns which are considered features
n_y = 22 # the column for label

# Load dataset
frames=[]
#data_file_path = "../data/featuredata_br/"
data_file_path = "../data/featuredata_br_copy/"
data_files = [f for f in listdir(data_file_path) if isfile(join(data_file_path, f)) and f.endswith('.csv')]
for i in data_files:
	url = data_file_path + i
	# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = pd.read_csv(url)#, names=names)
	frames.append(dataset)
	final_dataset = pd.concat(frames)

#Different buil-in functions of ndarray
print final_dataset.shape, final_dataset.ndim, len(final_dataset.shape) #, dataset.size#, dataset.dtype, dataset.itemsize

#Extracting the values from the dataframe
array = final_dataset.values

#Separating the features and the labels
array = final_dataset.values
#print array[0]
#Separating the features and the labels
X = array[:,1:]
Y = array[:,n_y]
qid = array[:,24]
#print X[0], Y[0], qid[0]

#Enable for labels

class_weight ={}

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

for s in set(Y):
	class_weight[s] = 90 - ord(s)

print "In whole data",Counter(Y)
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
	models.append(('SVM', LinearSVC(dual=False,class_weight = class_weight)))
	models.append(('dt', DecisionTreeClassifier(max_depth=4)))
	models.append(('rf',RandomForestClassifier(n_estimators=20)))
	#models.append(('CART', DecisionTreeClassifier()))
	models.append(('KNN', KNeighborsClassifier()))
	#models.append(('SVM', SVC(kernel='rbf', probability=True)))
	#models.append(('SVR', SVR(kernel='linear', C=1e3)))

except Exception, e:
	traceback.print_exc()
#Models_Evaluation
models_eval=[]
models_metrics=[]

#Evaluate the models
results = []
names = []
cv_predict=[]
for name, model in models:
	names.append(name)



# for i,value in enumerate(Y_test):
#  	if Y_test[i] < 1.0:
#  		Y_test[i] = 0.0
# Y_test = np.asarray(Y_test, dtype="f4")
k = 0
for i in range(0,len(models)):
	clf = models[i][1]
	clf.fit(X_train[:,k:-4], Y_train)
	pred = clf.predict(X_test[:,k:-4])
	ac_score = accuracy_score(Y_test, pred)
	print models[i][0],"Accuracy is ", ac_score
	print "In predicted data",Counter(pred)
	features = ['POS','NER','lemmas','dep_tree','F1','span_words', 'q_words', 'ground_truth','predicted_F1_score']
	combined_feature = []
	for j,item in enumerate(X_test[:,-8:]):
		combined_feature.append(list(item))
		combined_feature[j].append((pred[j]))

	df = pd.DataFrame.from_records(combined_feature, columns = features)
	output_file_path = "../data/predictions/classification/latest/"
	if not os.path.exists(output_file_path):
		os.makedirs(output_file_path)
	df.to_csv(os.path.join(output_file_path,models[i][0]+ str(k) +".csv"))

