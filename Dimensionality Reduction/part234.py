import numpy
import urllib
import scipy.optimize
import random
from math import exp
from math import log
import matplotlib.pyplot as plt

random.seed(0)

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
dataFile = open("winequality-white.csv")
header = dataFile.readline()
fields = ["constant"] + header.strip().replace('"','').split(';')
featureNames = fields[:-1]
labelName = fields[-1]
lines = [[1.0] + [float(x) for x in l.split(';')] for l in dataFile]
X = [l[:-1] for l in lines]
y = [l[-1] > 5 for l in lines]
print "done"

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

##################################################
# Logistic regression by gradient ascent         #
##################################################

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  # print "ll =", loglikelihood
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0]*len(theta)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
      dl[k] += X[i][k] * (1 - sigmoid(logit))
      if not y[i]:
        dl[k] -= X[i][k]
  for k in range(len(theta)):
    dl[k] -= lam*2*theta[k]
  return numpy.array([-x for x in dl])

X_train = X[:int(len(X)/3)]
y_train = y[:int(len(y)/3)]
X_validate = X[int(len(X)/3):int(2*len(X)/3)]
y_validate = y[int(len(y)/3):int(2*len(y)/3)]
X_test = X[int(2*len(X)/3):]
y_test = y[int(2*len(X)/3):]

##################################################
# Train                                          #
##################################################

def train(lam):
  theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, pgtol = 10, args = (X_train, y_train, lam))
  return theta

##################################################
# Predict                                        #
##################################################

def performance(theta):
  scores_train = [inner(theta,x) for x in X_train]
  scores_validate = [inner(theta,x) for x in X_validate]
  scores_test = [inner(theta,x) for x in X_test]

  predictions_train = [s > 0 for s in scores_train]
  predictions_validate = [s > 0 for s in scores_validate]
  predictions_test = [s > 0 for s in scores_test]

  correct_train = [(a==b) for (a,b) in zip(predictions_train,y_train)]
  correct_validate = [(a==b) for (a,b) in zip(predictions_validate,y_validate)]
  correct_test = [(a==b) for (a,b) in zip(predictions_test,y_test)]
  
  #trainErr = Errors(predictions_train, y_train)
  #ValidationErr = Errors(predictions_validate, y_validate)
  testErr = Errors(predictions_test, y_test)

  sigmoid_test_scores = [sigmoid(s) for s in scores_test]
  merge = zip(sigmoid_test_scores, y_test)	
  merge.sort(key=lambda x:x[0])
  relevant = sum(y_test)

  precision10, recall10 	= PrecisionRecall(merge, relevant, 10)
  print "10", precision10, recall10
  precision50, recall50 	= PrecisionRecall(merge, relevant, 500)
  print precision50, recall50
  precision1000, recall1000 = PrecisionRecall(merge, relevant, 1000)
  print precision1000, recall1000

  PlotPrecisionRecall(merge, relevant, len(y_test))

  acc_train = sum(correct_train) * 1.0 / len(correct_train)
  acc_validate = sum(correct_validate) * 1.0 / len(correct_validate)
  acc_test = sum(correct_test) * 1.0 / len(correct_test)
  return acc_train, acc_validate, acc_test, testErr

##################################################
# Errors                            #
##################################################

def Errors(prediciton, label):
  fpstvs = 0
  fngtvs = 0
  tpstvs = 0
  tngtvs = 0

  for i in xrange(len(label)):
    if label[i] == True and prediciton[i] == True:
      tpstvs = tpstvs + 1
    if label[i] == False and prediciton[i] == True:
      fpstvs = fpstvs + 1
    if label[i] == True and prediciton[i] == False:
      fngtvs = fngtvs + 1
    if label[i] == False and prediciton[i] == False:
      tngtvs = tngtvs + 1      

  ber = 0.5*(fpstvs*1.0/(fpstvs+tngtvs) + fngtvs*1.0/(fngtvs+tpstvs))

  return (tpstvs, tngtvs, fpstvs, fngtvs, ber)    

##################################################
# Precision & Recall                            #
##################################################

def PrecisionRecall(merge, relevant, retrieved):
	count = 0
	x = len(merge) - 1
	while x>=0:
		if len(merge) - x > retrieved:
			break
		t = merge[x]	
		if t[0] > 0.5 and t[1] == True:
			count = count + 1
		x = x -1		
	precision = count*1.0/retrieved
	recall = count*1.0/relevant		
	return (precision, recall)

##################################################
# Precision & Recall Graph                          #
##################################################

def PlotPrecisionRecall(merge, relevant, retrieved):
	count = 0
	result = []
	x = len(merge) - 1
	while x>=0:
		t = merge[x]	
		if t[0] > 0.5 and t[1] == True:
			count = count + 1
		precision = count*1.0/(len(merge) - x)
		recall = count*1.0/relevant		
		result.append((precision, recall))
		x = x - 1
	p,r = zip(*result)
	plt.scatter(r,p)
	#plt.show()

##################################################
# Validation pipeline                            #
##################################################

for lam in [0.01]:
  theta = train(lam)
  acc_train, acc_validate, acc_test, testErr = performance(theta)
  print("lambda = " + str(lam) + ";\ttrain=" + str(acc_train) + "; validate=" + str(acc_validate) + "; test=" + str(acc_test))
  print "test err : ",testErr
