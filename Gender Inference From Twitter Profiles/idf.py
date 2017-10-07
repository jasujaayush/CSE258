import numpy
import scipy.optimize
import random
from math import exp
from math import log
import csv
import nltk
import string
from nltk.stem.porter import *
from collections import defaultdict
from nltk.corpus import stopwords
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

dataFile = open("data.csv", 'rU')
reader = csv.reader(dataFile)
headers = reader.next()
data =[]
for row in reader:
    d = {}
    for i in xrange(len(headers)):
        d[headers[i]] = row[i]
    if d['gender:confidence'] == '1' and d['gender'] in ('male','female','brand'):
        data.append(d)

train = data[:11000]
test = data[11000:]

punctuation = set(string.punctuation)
stopWords = set(stopwords.words('english'))
wordCount = defaultdict(int)
for d in train:
  r = ''.join([c for c in (d['text'] + " " + d['description'] + " " + d['sidebar_color']).lower() if not c in punctuation])
  for w in r.split():
    if w not in stopWords:
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()
words = [x[1] for x in counts[:1000]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

idf = defaultdict(float)
for d in train:
  r = ''.join([c for c in (d['text'] + " " + d['description'] + " " + d['sidebar_color']).lower() if not c in punctuation])
  for w in set(r.split()):
    if (w not in stopWords) and (w in wordSet):
        idf[w] += 1

n = len(train)
for k in idf:
  idf[k] = math.log(idf[k]*1.0/n)/math.log(10)           

def feature(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in (datum['text'] + " " + datum['description'] + " " + datum['sidebar_color']).lower() if not c in punctuation])
  for w in r.split():
    if w in idf:
      feat[wordId[w]] += idf[w]
  return feat

gender = {}
gender['male'] = 1
gender['female'] = -1
gender['brand'] = 0
X_train = [feature(d) + [int(d['tweet_count']) > 68000] for d in train]
Y_train = [gender[d['gender']] for d in train]
X_test = [feature(d) + [int(d['tweet_count']) > 68000] for d in test]
Y_test = [gender[d['gender']] for d in test]

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, Y_train) 
print logreg.score(X_test, Y_test) 



nb = MultinomialNB()
nb.fit(X_train, Y_train)
print nb.score(X_test, Y_test)