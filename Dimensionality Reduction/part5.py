import numpy
import urllib
import scipy.optimize
import random
from sklearn.decomposition import PCA
from math import exp
from math import log

random.seed(0)

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
dataFile = open("winequality-white.csv")
header = dataFile.readline()
fields = header.strip().replace('"','').split(';')
featureNames = fields[:-1]
labelName = fields[-1]
lines = [[float(x) for x in l.split(';')] for l in dataFile]
X = [l[:-1] for l in lines]
y = [l[-1] for l in lines]
print "done"

##################################################
# Mean Feature                                          #
##################################################

def MeanFeature(data):
  data = numpy.array(data)
  avg = sum(data)*1.0/len(data)
  meandata = [avg for x in xrange(len(data))]
  return meandata

def CostMeanFeature(data):
  data = numpy.array(data)
  avg = sum(data)*1.0/len(data)
  meandata = [avg for x in xrange(len(data))]
  error = data - meandata
  error = error*error
  return sum(sum(error))

def ReconstructionError(data,d):
  data = numpy.array(data)
  data = data.T
  data = data[d:]
  data = data.T
  avg = sum(data)*1.0/len(data)
  meandata = [avg for x in xrange(len(data))]
  error = data - meandata
  error = error*error
  return sum(sum(error))


X_train = X[:int(len(X)/3)]
print "Cost of mean feature : ", CostMeanFeature(X_train)
y_train = y[:int(len(y)/3)]
X_validate = X[int(len(X)/3):int(2*len(X)/3)]
y_validate = y[int(len(y)/3):int(2*len(y)/3)]
X_test = X[int(2*len(X)/3):]
y_test = y[int(2*len(X)/3):]

pca = PCA(n_components=len(X[0]))
pca.fit(X_train)
pc = pca.components_
print pc
pcx_train = [pc.dot(t) for t in X_train]
pcx_test = [pc.dot(t) for t in X_test]
#print "Reconstruction Error (code)    : ", ReconstructionError(pcx_train,4)
print "Reconstruction Error (Library) : ", sum(pca.explained_variance_[4:])*len(pcx_train)  

#Adding bias term in the feature
column = numpy.ones((len(pcx_train),1))
data_train = numpy.concatenate((column, numpy.mat(pcx_train)), axis=1)
#data_train = [[1.] + d for d in pcx_train]
column = numpy.ones((len(pcx_test),1))
data_test = numpy.concatenate((column, numpy.mat(pcx_test)), axis=1)
#data_test  = [[1.] + d for d in pcx_test]

data_train = data_train.tolist()
data_test = data_test.tolist()
print data_train[0]
for x in xrange(2, len(data_train[0]) + 1):
  feature_train = numpy.array([d[:x] for d in data_train])
  feature_test = numpy.array([d[:x] for d in data_test])
  theta,residuals_train,rank,s = numpy.linalg.lstsq(feature_train, y_train)
  #Running on testTest data 
  output = feature_test.dot(theta)
  output = output - numpy.array(y_test) #(x.theta - y)
  test_mse = output.dot(output) # squared sum for mse
  print "features : ", x-1, " Train MSE : ", residuals_train*1.0/len(y_train), " Test MSE : ",test_mse/len(y_test)