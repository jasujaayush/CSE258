import numpy
import scipy.optimize
import urllib
import random
import csv
from sklearn import svm

def parseData(fname):
  data = []
  with open(fname) as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=';')
    for row in spamreader:
      data.append(row)
    return data

print "Reading data..."
data = parseData("winequality-white.csv")
print "done"

h = len(data)/2
train = data[:h]
test = data[h+1:]

def feature(datum,keys):
  f = [float(datum[key]) for key in keys]
  feat = [1] + f
  return feat

#training the model
keys = [k for k in data[0]]
keys.remove('quality')
x_train = [feature(d,keys) for d in train]
y_train = [float(d['quality']) > 5 for d in train]
clf = svm.SVC(C=100)
clf.fit(x_train, y_train)

x_test = [feature(d,keys) for d in test]
y_test = [float(d['quality']) > 5 for d in test]

train_predictions = clf.predict(x_train)
test_predictions = clf.predict(x_test)

correct_train = (numpy.array(train_predictions) == numpy.array(y_train))
print "Accuracy_train = ", sum(correct_train)*1.0/len(correct_train)

correct_test = (numpy.array(test_predictions) == numpy.array(y_test))
print "Accuracy_test = ", sum(correct_test)*1.0/len(correct_test)