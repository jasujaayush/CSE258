import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict
import nltk
import math
import string
from nltk.stem.porter import *
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

### Just the first 5000 reviews

print "Reading data..."
data = list(parseData("beer_50000.json"))[:5000]
print "done"

### Just take the most popular words...

idf = defaultdict(float)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in set(r.split()):
    idf[w] += 1

n = len(data)
for k in idf:
  idf[k] = math.log(idf[k]*1.0/n)/math.log(10)

t = ['foam', 'smell', 'banana', 'lactic', 'tart']
tidf = defaultdict(float)
for w in t:
  print "idf " + w + " : " + str(idf[w])

d = data[0]
r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
for w in r.split():
  if w in t:
    tidf[w] += idf[w]
print tidf    

words = [x for x in idf]
wordId = dict(zip(words, range(len(words))))

def feature(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  for w in r.split():
    if w in idf:
      feat[wordId[w]] += idf[w]
  return feat

f0 = numpy.array(feature(data[0]))
f1 = numpy.array(feature(data[1]))
print "cosine similariy : " + str(f0.dot(f1))

similar = 1
similarity = f0.dot(f1)
for x in xrange(2, len(data)):
  f = numpy.array(feature(data[x]))
  s = f0.dot(f)
  if s >= similarity:
    similarity = s
    similar = x + 1

print "beer most similar to first : " + data[similar - 1]['beer/beerId']


# ...................part 8.........................

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()
words = [x[1] for x in counts[:1000]]
wordId = dict(zip(words, range(len(words))))

idf = defaultdict(float)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in set(r.split()):
    if w in words:
      idf[w] += 1

n = len(data)
for k in idf:
  idf[k] = math.log(idf[k]*1.0/n)/math.log(10)


def feature8(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  for w in r.split():
    if w in words:
      feat[wordId[w]] += idf[w]
  feat.append(1) #offset
  return feat

X = [feature8(d) for d in data]
y = [d['review/overall'] for d in data]

clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_
predictions = clf.predict(X)
print "MSE with idf of 1000 most frequent unigrams" + str(mean_squared_error(predictions,y))