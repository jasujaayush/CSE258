import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict
import nltk
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

def getBigrams(wordlist):
  l = []  			
  for i in xrange(len(wordlist)-1):
    s = wordlist[i] + " " + wordlist[i+1]
    l.append(s)
  return l 

### Ignore capitalization and remove punctuation

bigram = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  wordlist = r.split()
  blist = getBigrams(wordlist)
  for s in blist:
    bigram[s] += 1

bcounts = [(bigram[w], w) for w in bigram]
bcounts.sort()
bcounts.reverse()
print len(bcounts)
print bcounts[:5]

bigrams = [x[1] for x in bcounts[:1000]]

bigramId = dict(zip(bigrams, range(len(bigrams))))
bigramSet = set(bigrams)

def feature(datum):
  feat = [0]*len(bigrams)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  wordlist = r.split()
  blist = getBigrams(wordlist)
  for b in blist:
    if b in bigrams:
      feat[bigramId[b]] += 1
  feat.append(1) #offset
  return feat

X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]

clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_
predictions = clf.predict(X)
print "MSE " + str(mean_squared_error(predictions,y))