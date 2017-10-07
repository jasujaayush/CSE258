import numpy
import urllib
import scipy.optimize
import random
import csv
from math import exp
from math import log

def parseData(fname):
  data = []
  with open(fname) as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=';')
    for row in spamreader:
      data.append(row)
    return data

def feature(datum,keys):
  f = [float(datum[key]) for key in keys]
  feat = [1] + f
  return feat

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + numpy.exp(-x))

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= numpy.log(1 + numpy.exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  print "ll =", loglikelihood
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = numpy.array([0.0]*len(theta))
  # Fill in code for the derivative
  for i in range(len(X)):
    x = numpy.array(X[i])
    swx = sigmoid(inner(theta,x))
    dl = dl + (y[i] - swx)*x + 2*lam*numpy.array(theta)
  # Negate the return value since we're doing gradient *ascent*
  return numpy.array([-x for x in dl])  

print "Reading data..."
data = parseData("winequality-white.csv")
print "done"

h = len(data)/2
train = data[:h]
test = data[h:]

keys = [k for k in data[0]]
keys.remove('quality')
X_train= [feature(d,keys) for d in train]
y_train = [float(float(d['quality']) > 5) for d in train]

X_test= [feature(d,keys) for d in test]
y_test = [float(float(d['quality']) > 5) for d in test]

# If we wanted to split with a validation set:
#X_valid = X[len(X)/2:3*len(X)/4]
#X_test = X[3*len(X)/4:]

'''
theta = [0.005]*len(X_train[0])
lam = .001
n = .00000012
l = 0
while True:
  l_new = f(theta, X_train, y_train, lam)
  if abs(l - l_new) >= .001:
    l = l_new
    theta = theta - n*fprime(theta, X_train, y_train, lam)
  else:
    break    
'''

# Use a library function to run gradient descent (or you can implement yourself!)
theta,l,info = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X_train[0]), fprime, args = (X_train, y_train, .0005))
print "Final log likelihood = ", -l

X_test = numpy.array(X_test)
y_test = numpy.array(y_test)
y_cap = [sigmoid(wx) for wx in X_test.dot(theta)]
y_cap = numpy.array([float(t > 0.5) for t in y_cap])
compared_result = (y_cap == y_test)
print len(compared_result), sum(compared_result)
print "Accuracy = ",sum(compared_result)*1.0/len(compared_result)
