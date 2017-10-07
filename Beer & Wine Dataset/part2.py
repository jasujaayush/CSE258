import numpy
import urllib
import scipy.optimize
import random
import csv

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
x = [feature(d,keys) for d in train]
y = [float(d['quality']) for d in train]
theta,residuals,rank,s = numpy.linalg.lstsq(x, y)
print "theta : ", theta
print "Training MSE : ", residuals/len(train)

#running it on test data now.
x = [feature(d,keys) for d in test]
y = [float(d['quality']) for d in test]
x = numpy.array(x)
y = numpy.array(y)
output = x.dot(theta)
output = output - y #(x.theta - y)
test_mse = output.dot(output) # squared sum for mse
print "Test MSE : ",test_mse/len(test)

extra = {}
for t in keys:
  temp = keys[:]
  temp.remove(t)
  x = [feature(d,temp) for d in train]
  y = [d['quality'] for d in train]
  coeff,res,rank,s = numpy.linalg.lstsq(x, y)

  #running it on test data now.
  x = [feature(d,temp) for d in test]
  y = [float(d['quality']) for d in test]
  x = numpy.array(x)
  y = numpy.array(y)
  output = x.dot(coeff)
  output = output - y #(x.theta - y)
  test_mse_2 = output.dot(output) # squared sum for mse
  print t + " " + str(test_mse_2/len(test))
  extra[t] = test_mse_2 - test_mse # the one whose removal gives the most MSE increase, provides the maximum additional info

print "Extra : ", extra


