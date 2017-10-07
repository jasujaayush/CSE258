import numpy
import urllib
import scipy.optimize
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("train.json"))
print "done"

train = data[:100000]
test = data[100000:]

def getFeaturesAndRatings(data):
	ratings = []
	X = []
	for d in data:
		rating = d['rating']
		X.append([1.0])
		ratings.append(rating)
	return X, ratings	

def getPredictedRatings(data, theta):
	predictions = []
	true = []
	for d in data:
		rating = d['rating']
		predictions.append(theta)
		true.append(rating)
	return predictions, true

X_train, Y_train = getFeaturesAndRatings(train)
theta,residuals,rank,s = numpy.linalg.lstsq(X_train, Y_train)

print "alpha : ", theta[0]
Y_predictions, Y_true = getPredictedRatings(test, theta)
mae1 = mean_squared_error(Y_true, Y_predictions)	
print "Q5. MSE : " +  str(mae1)


def getFeatures(data):
	alpha = 5*random.random()
	betau = {}
	betai = {}
	itemsWithUser = {}
	usersWithItem = {}
	ratings = defaultdict(dict)
	for d in data:
		itemid = d['itemID']
		userid = d['reviewerID']
		rating = d['rating']
		betau[userid] = 5*random.random();
		betai[itemid] = 5*random.random();

		if userid in itemsWithUser:
			itemsWithUser[userid].append(itemid)
		else:
			itemsWithUser[userid] = [itemid]

		if itemid in usersWithItem:
			usersWithItem[itemid].append(userid)
		else:
			usersWithItem[itemid] = [userid]

		ratings[userid][itemid] = rating
	return alpha, betau, betai, itemsWithUser, usersWithItem, ratings	

def trainParam(alpha, betau, betai, itemsWithUser, usersWithItem, ratings, lmbda = 1):	
	N = 0
	alpha = 0;
	for user in ratings:
		for item in ratings[user]:
			alpha = alpha + (ratings[user][item] - betau[user] - betai[item])
			N = N + 1
	alpha = alpha/N

	for user, items in itemsWithUser.iteritems():
		betaUpdate = 0
		l = len(items)
		for item in items:
			betaUpdate = betaUpdate + (ratings[user][item] - alpha - betai[item])
		betau[user]	= betaUpdate/(lmbda + l)

	for item, users in usersWithItem.iteritems():
		betaUpdate = 0
		l = len(users)
		for user in users:
			betaUpdate = betaUpdate + (ratings[user][item] - alpha - betau[user])
		betai[item]	= betaUpdate/(lmbda + l)

	return alpha

def getTestFeatures(data):
	features = []
	for d in data:
		feature = []	
		itemid = d['itemID']
		userid = d['reviewerID']
		rating = d['rating']
		feature.append(userid)
		feature.append(itemid)
		feature.append(rating)
		features.append(feature)
	return features

def validation(features, alpha, betau, betai):	
	predictions = []
	true = []
	for feature in features:
			prediction = alpha
			if feature[0] in betau:
				prediction = prediction + betau[feature[0]]
			if feature[1] in betai:	 
			  	prediction = prediction + betai[feature[1]]
			predictions.append(prediction)
			true.append(feature[-1])
	predictions = numpy.array(predictions)
	true = numpy.array(true)
	return mean_squared_error(true, predictions)			


for lmbda in xrange(5,6):
	alpha, betau, betai, itemsWithUser, usersWithItem, ratings = getFeatures(train)
	features = getTestFeatures(test)
	prev = 100;
	while(True):
		mse = validation(features, alpha, betau, betai)
		if prev - mse < .000000001:
			break
		prev = mse	
		alpha = trainParam(alpha, betau, betai, itemsWithUser, usersWithItem, ratings,lmbda)
	print lmbda, mse
	if lmbda == 1:
		maxbetauser = ''
		temp = -1
		for user in betau:	
			if betau[user] > temp:
				maxbetauser = user
				temp = betau[user]

		minbetauser = ''
		temp = 1000
		for user in betau:	
			if betau[user] < temp:
				minbetauser = user
				temp = betau[user]

		maxbetaitem = ''
		temp = -1
		for item in betai:	
			if betai[item] > temp:
				maxbetaitem = item
				temp = betai[item]

		minbetaitem = ''
		temp = 1000
		for item in betai:	
			if betai[item] < temp:
				minbetaitem = item
				temp = betai[item]


		print maxbetauser + " : " + str(betau[maxbetauser]) + " " + minbetauser + " : " + str(betau[minbetauser])
		print maxbetaitem + " : " + str(betai[maxbetaitem]) + " " + minbetaitem + " : " + str(betai[minbetaitem])
		print maxbetauser in ratings
		print minbetauser in ratings

	if lmbda == 5:
		predictions = open("predictions_Rating.txt", 'w')
		for l in open("pairs_Rating.txt"):
		  if l.startswith("userID"):
		    #header
		    predictions.write(l)
		    continue
		  u,i = l.strip().split('-')
		  prediction = alpha
		  if u in betau:
			prediction = prediction + betau[u]
		  if i in betai:	 
			prediction = prediction + betai[i]
		  predictions.write(u + '-' + i + ',' + str(prediction) + '\n')
		predictions.close()



