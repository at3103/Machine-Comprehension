import numpy
import scipy
import pandas
import matplotlib	
import sklearn
import sys
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
#from sklearn import cross_validation
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#To check the version of python
#print "Python : {}".format(sys.version)

# Load dataset
url = "iris1.csv"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url)#, names=names)


#Different buil-in functions of ndarray
print dataset.shape, dataset.ndim, len(dataset.shape), dataset.size#, dataset.dtype, dataset.itemsize
'''
#To take a quick peek
print dataset.head(20)

#To print summary
print dataset.describe()

#To know the different classes of the dataset
print dataset.groupby('Species').size()
print dataset.groupby('Sepal.Width').size()

#Data Visualization:  Plots

#Box Plot
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

#Histogram
dataset.hist()
plt.show()


#Scatter Plot
scatter_matrix(dataset)
plt.show()
'''

'''
Extracting test and train data sets from given data set
'''

#Extracting the values from the dataframe
array = dataset.values

#Separating the features and the labels
X = array [:,0:4]
Y = array[:,4]

#Validation Size
test_size = 0.2

#Set the seed for randomness here
seed = 7

#Obtain the training and test sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,
	test_size = test_size, random_state = seed)


'''
Preparing the metrics for evaluation and cross-validation
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
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))

#Models_Evaluation
models_eval=[]
models_metrics=[]

#Evaluate the models
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=n_fold, random_state=cv_seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=10, scoring=scoring)
	#cv_results1= model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	models_eval.append(cv_results.mean())
	print name + " : ", cv_results.mean(),"(",cv_results.std(),")"

for i in range(0,len(names)):
	models[i][1].fit(X_train, Y_train) 
	predictions = models[i][1].predict(X_test)
	ac_score = accuracy_score(Y_test, predictions)
	conf_matrix = confusion_matrix(Y_test, predictions)
	class_rep = classification_report(Y_test, predictions)	
	models_metrics.append([names[i],ac_score, conf_matrix, class_rep])
	print names[i],models_eval[i], ac_score

for i in models_metrics:
	print i[0], i[1]
	print i[2]
	print i[3]




#a = numpy.array([(1,2,3),(2,3,1)])
#Different buil-in functions of ndarray
#print a.ndim, a.shape, len(a.shape), a.size, a.dtype, a.itemsize
#print a.data
#b = numpy.arange(20).reshape(10,2)


'''
Reference :	1. http://machinelearningmastery.com/machine-learning-in-python-step-by-step/
			2. http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
'''