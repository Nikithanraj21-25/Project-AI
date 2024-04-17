from pandas import read_csv #for handling csv files

from pandas.plotting import scatter_matrix
from matplotlib import pyplot

#loading library for algorithm, splitting dataset, evaluation method
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split #for splitting dataset into train and test

#for evaluate algorithms
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

fileName = "Energy Meter.csv"
names = ['Voltage', 'Current', 'Power', 'Class']
dataset = read_csv(fileName, names = names)
print(dataset)

#Summarize dataset
print(dataset.shape) #show no. of rows and columns
print(dataset.head(5)) #shows first five data
print(dataset.describe()) #Details of dataset
print(dataset.groupby('Class').size()) #Count dataset based on classes

#visualize dataset
#BAR PLOT
ax = dataset.plot(kind='bar', subplots=True, layout=(2, 2), use_index=False)
for row in ax:
    for subplot in row:
        subplot.grid(False)  # Remove gridlines
        subplot.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove x-axis ticks and labels
pyplot.suptitle('BAR PLOT')
pyplot.show()

#HISTOGRAM
dataset.hist()
pyplot.suptitle('HISTOGRAM PLOT')
pyplot.show()

#SCATTER PLOT
scatter_matrix(dataset)
pyplot.suptitle('SCATTER PLOT')
pyplot.show()

#Segregation of data as input and output
array = dataset.values #store dataset in array
x = array[:,0:3] #input
y = array[:,3] #output
print(x)
print(y)

#Splitting dataset for training and validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=1, shuffle=True)

#Loading ML Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
res = []
mean_accuracy = []
model_names = []
for name, model in models:
  kfold = StratifiedKFold(n_splits=10, random_state=None)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  res.append(cv_results.mean())
  print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
  mean_accuracy.append(cv_results.mean())
  model_names.append(name)

#Accuracy Comparison Graph between Algorithms
pyplot.ylim(.990, .999)
pyplot.bar(model_names, mean_accuracy, color='red', width=0.6)
pyplot.suptitle('Algorithm Comparison')
pyplot.show()
