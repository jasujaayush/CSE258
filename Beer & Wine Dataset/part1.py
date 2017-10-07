import numpy
import urllib
import scipy.optimize
import random

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("beer_50000.json"))
print "done"

def feature1(datum):
  f = datum['review/timeStruct']['year']	
  feat = [1] + [f]
  return feat

def UniqueYears(data):
  y = [d['review/timeStruct']['year'] for d in data] 		
  yearSet = set(y)	
  return list(yearSet)

def feature2(datum, lenYears, minYear):
  f = datum['review/timeStruct']['year'] - minYear	
  feat = [0]*lenYears
  feat[f] = 1
  feat = [1] + feat
  return feat 

years = UniqueYears(data)
minYear = min(years)
lenYears = len(years)
print lenYears
print minYear, max(years)
x2 = [feature2(d, lenYears, minYear) for d in data]
x1 = [feature1(d) for d in data]
y = [d['review/overall'] for d in data]
theta1,residuals1,rank,s = numpy.linalg.lstsq(x1, y)
theta2,residuals2,rank,s = numpy.linalg.lstsq(x2, y)

print theta1
print "1> MSE : ",residuals1*1.0/(len(x1))
print "2> MSE : ",residuals2*1.0/(len(x2))

