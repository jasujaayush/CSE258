import numpy
import urllib
import scipy.optimize
from sklearn.metrics import mean_absolute_error
import random
from sklearn import svm

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("train.json"))
print "done"

train = data[:100000]
test = data[100000:]

def getFeaturesAndRatios(data):
	ratios = []
	X = []
	for d in data:
		o = d['helpful']['outOf']
		if(o != 0):
			h = d['helpful']['nHelpful']
			X.append([1.0])
			ratio = float(h)/o
			ratios.append(ratio)
	return X, ratios	

def getHelpfulReviews(data, theta):
	features = []
	true = []
	for d in data:
		o = d['helpful']['outOf']
		features.append(o)
		h = d['helpful']['nHelpful']
		true.append(h)
	features = numpy.array(features)
	predictions = theta*features
	return predictions, true

X_train, Y_train = getFeaturesAndRatios(train)
theta,residuals,rank,s = numpy.linalg.lstsq(X_train, Y_train)

print "alpha : " + str(theta[0])

Y_predictions, Y_true = getHelpfulReviews(test, theta)
mae = mean_absolute_error(Y_true, Y_predictions)	

print "MAE : " + str(mae)

def getFeaturesAndRatios3(data):
	ratios = []
	X = []
	for d in data:
		o = d['helpful']['outOf']
		if(o != 0):
			h = d['helpful']['nHelpful']
			review = d['reviewText'].strip('\n\t').split()
			rating = d['rating']
			X.append([1.0] + [len(review), rating])
			ratio = float(h)/o
			ratios.append(ratio)
	return X, ratios

def getHelpfulReviews3(data, theta):
	features = []
	true = []
	for d in data:
		o = d['helpful']['outOf']
		review = d['reviewText'].strip('\n\t').split()
		rating = d['rating']
		feature = numpy.array([1.0] + [len(review), rating])
		feature = o*feature
		features.append(feature)
		h = d['helpful']['nHelpful']
		true.append(h)
	features = numpy.array(features)
	predictions = [f[0]*theta[0] + f[1]*theta[1] + f[2]*theta[2] for f in features]
	return predictions, true		

def getTestHelpfulReviews4(data, theta):
	features = []
	details = "userID-itemID-outOf,prediction\n"
	for d in data:
		userid = d['reviewerID']
		itemid = d['itemID']
		o = d['helpful']['outOf']
		detail = userid + '-' + itemid + '-' + str(o) + ','

		review = d['reviewText'].strip('\n\t').split()
		rating = d['rating']
		feature = numpy.array([1.0] + [len(review), rating])
		feature = o*feature
		prediction = feature[0]*theta[0] + feature[1]*theta[1] + feature[2]*theta[2]
		detail = detail + str(prediction) + "\n"
		details = details + detail
	return details

X_train, Y_train = getFeaturesAndRatios3(train)
theta,residuals,rank,s = numpy.linalg.lstsq(X_train, Y_train)

print "params : ", str(theta)

Y_predictions, Y_true = getHelpfulReviews3(test, theta)
mae = mean_absolute_error(Y_true, Y_predictions)	

print "mae : ", mae

test_data = list(parseData("test_Helpful.json"))
predDetails = getTestHelpfulReviews4(test_data, theta)

text_file = open("pairs_Helpful.txt", "w")
text_file.write(predDetails)
text_file.close()

