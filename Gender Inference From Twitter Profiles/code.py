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
      r = ''.join([c for c in d['text'].lower() if not c in punctuation])
      for w in r.split():
        if w not in stopWords:
            wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()
words = [x[1] for x in counts[:1000]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)  

def feature(d):
    feat = [0]*len(words)
    r = ''.join([c for c in d['text'].lower() if not c in punctuation])
    for w in r.split():
        if w in words:
            feat[wordId[w]] = 1
    feat.append(1) #offset
    return feat

wordCountDes = defaultdict(int)
for d in train:
    r = ''.join([c for c in d['description'].lower() if not c in punctuation])
    for w in r.split():
        if w not in stopWords:
            wordCountDes[w] += 1
    wordCountDes[d['sidebar_color']] += 1

counts = [(wordCountDes[w], w) for w in wordCountDes]
counts.sort()
counts.reverse()
wordsDes = [x[1] for x in counts[:1000]]
wordIdDes = dict(zip(wordsDes, range(len(wordsDes))))
wordSetDes = set(wordsDes)

def featureDes(d):
    feat = [0]*len(words)
    r = ''.join([c for c in d['description'].lower() if not c in punctuation])
    for w in r.split():
        if w in wordsDes:
            feat[wordIdDes[w]] = 1
    color = d['sidebar_color']        
    if color in wordsDes:
        feat[wordIdDes[color]] = 1
    feat.append(1) #offset
    return feat    

def featureCount(d):
    feat = []
    feat.append(int(d['tweet_count']))
    feat.append(1) #offset
    return feat 

colors = []
for d in train:
    colors.append(d['sidebar_color'])
colorset = set(colors)
colorId = dict(zip(colorset, range(len(colorset))))
def featureSidebar(d):
    feat = [0]*len(colorset)
    c = d['sidebar_color']
    if c in colorset:
        feat[colorId[c]] = 1
    feat.append(1)
    return feat     

gender = {}
gender['male'] = 1
gender['female'] = -1
gender['brand'] = 0
X_train = [featureDes(d) + feature(d) + [int(d['tweet_count']) > 68000] for d in train]
Y_train = [gender[d['gender']] for d in train] 
X_test = [featureDes(d) + feature(d) + [int(d['tweet_count']) > 68000] for d in test]
Y_test = [gender[d['gender']] for d in test]

X_train = [featureSidebar(d) for d in train]
Y_train = [gender[d['gender']] for d in train]
X_test = [featureSidebar(d) for d in test]
Y_test = [gender[d['gender']] for d in test]

nb = MultinomialNB()
nb.fit(X_train, Y_train)
print nb.score(X_test, Y_test)

probs_train = nb.predict_proba(X_train)
probs_test  = nb.predict_proba(X_test)
out = '_unit_id , male_score , female_score , brand_Score\n'
for i in xrange(len(train)):
    s = train[i]['_unit_id'] +  " , " + str(probs_train[i][2]) + " , " + str(probs_train[i][0]) + " , " + str(probs_train[i][1]) + '\n'
    out += s    

for i in xrange(len(test)):
    s = test[i]['_unit_id'] +  " , " + str(probs_test[i][2]) + " , " + str(probs_test[i][0]) + " , " + str(probs_test[i][1]) + '\n'
    out += s        

classifier = Pipeline([('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(X_train, Y_train)
print classifier.score(X_test, Y_test)   

clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X_train, Y_train)
theta = clf.coef_
Y_pred = clf.predict(X_test)
predictions = []
for p in Y_pred:
    predictions.append(round(p))

correct = 0
for i in xrange(len(predictions)):
    if predictions[i] == Y_test[i]:
        correct += 1
print correct*1.0/len(predictions)

'''
# plot histogram of frequency
l = counts[:50]
f = [t[0] for t in l] 
w = [t[1].encode('latin') for t in l]
pos = np.arange(len(w)) 
width=1.0
ax = plt.axes()                  
ax.set_xticks(pos + (width / 2))                          
ax.set_xticklabels(w,rotation=45)
plt.bar(pos, f, width, color='b')
plt.show() 

def parseData(fname):       
for l in urllib.urlopen(fname): 
yield eval(l)   

wordCount_female = defaultdict(int)
for d in data:
if d['gender'] == 'female':
r = ''.join([c for c in d['text'].lower() if not c in punctuation])
for w in r.split():
if w not in stopWords:
wordCount_female[w] += 1

counts = [(wordCount_female[w], w) for w in wordCount_female]
counts.sort()
counts.reverse()
words_female = [x[1] for x in counts[:1000]]  

wordCount_male = defaultdict(int)
for d in data:
if d['gender'] == 'male':
r = ''.join([c for c in d['text'].lower() if not c in punctuation])
for w in r.split():
if w not in stopWords:
wordCount_male[w] += 1

counts = [(wordCount_male[w], w) for w in wordCount_male]
counts.sort()
counts.reverse()
words_male = [x[1] for x in counts[:1000]]
'''