#!/usr/bin/python

import sys
import csv
import math
import time
import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

TRAINING_DATA = 'data/knn_train.csv'
TESTING_DATA = 'data/knn_test.csv'
DEBUG = 1

class Matrix:
	def __init__(self, file):
		self.data = self.__read_data(file)
		self.__row = len(self.data)
		self.__col = len(self.data[1])
		self.__getMatrix()

	def __read_data(self, file):
		with open(file, 'r') as f:
			data = [row for row in csv.reader(f.read().splitlines())]
    		return np.array(data)
	
	def __getMatrix(self):
		self.featureM = self.data[:,1:].astype(np.float)
		self.trueM = self.data[:,0]

	def normalize(self):
		maxOfCol = []
		minOfCol = []

		rowN = len(self.featureM)
		colN = len(self.featureM[0])

		# axis = 0 count for col of the array
		maxOfCol = np.max(self.featureM,axis=0)
		minOfCol = np.min(self.featureM,axis=0)

		for i in range(rowN):
		    for j in range(colN): # pass the first column
		        num = self.featureM[i][j]
		        minV = minOfCol[j]
		        maxV = maxOfCol[j]
		        self.featureM[i][j] = (num - minV)/(maxV - minV)
		        print self.featureM[i][j]




class K_nearest:
	def __init__ (self, trainM, testM):
		# training data
		self.train_featureM = trainM.featureM
		self.train_trueM 	= trainM.trueM

		# testing data
		self.test_featureM 	= testM.featureM
		self.test_trueM 	= testM.trueM

		self.__row 	= len(self.train_featureM)
		self.__col 	= len(self.train_featureM[0])
		self.error	= 0

 	def __euclideanDist(self,x,xi):
 		d = 0.0
 		for i in range(len(x)):
 			d += pow( (float(x[i]) - float(xi[i])) , 2)
 		return math.sqrt(d)


 	def getNeighbors(self,cur_trainInstance,k):
 		distanceSet = []
 		sortedSet 	= []
 		nearest 	= []
 		nearestIndex = []

		for trainRow in self.train_featureM:		# other training instances
			d = self.__euclideanDist(cur_trainInstance,trainRow)
			distanceSet.append(round(d,5))
		
		sortedSet = distanceSet[:]	# for sorting purpose
		sortedSet.sort()

		for i in range(1,k+1):	# pass the first value since it's the distance of itself ( = 0)
			nearest.append(sortedSet[i])

		for i in nearest:
			nearestIndex.append(distanceSet.index(i))

		#return the index# of all nearest neighbour of current training instance
		return nearestIndex


	def getVotes(self,k,d):
		if d == "train":
			featureM = self.train_featureM
			trueM = self.train_trueM
		else:	# 'test'
			featureM = self.test_featureM
			trueM = self.test_trueM

		# #get vote for each row of learning target matrix
		# for i in range(len(featureM)):
		# 	match = 0

		# 	instance = featureM[i]
		# 	truth = trueM[i]

		# 	nearestIndex = self.getNeighbors(instance,k)
		# 	voteSet = trueM[nearestIndex]

		# 	for vote in voteSet:	# k elements in this voteset
		# 		if vote == truth:
		# 			match += 1	# result matched 
		# 		else:
		# 			match -= 1	# not matched 

		# 	if match < 0:
		# 		self.error += 1


		if (DEBUG):
			match = 0

			instance = self.train_featureM[56]
			truth = self.train_trueM[56] 

			nearestIndex = self.getNeighbors(instance,k)
			voteSet = self.train_trueM[nearestIndex]

			for vote in voteSet:	# k elements in this voteset
				if vote == truth:
					match += 1	# result matched 
				else:
					match -= 1	# not matched 

			if match < 0:
				self.error += 1

			print "Nearest Idx:",nearestIndex
			print "vote sets:",voteSet
			print "truth:",truth
			print "match:",match
			print "error:",self.error
			print "----------------------------------"
		return self.error

	def reset(self):
		self.error	= 0








if __name__ == "__main__":
	trainM = Matrix(TRAINING_DATA) 
	testM = Matrix(TESTING_DATA) 

	K_nearest(trainM,testM).getVotes(21,'train')



	# trainError = []
	# testError = []

	# trainRowNum = float(len(trainM.featureM))
	# testRowNum = float(len(testM.featureM))

	# oddNumList = range(1,61,2) # range of 59

	# for k in oddNumList:
	# 	testInst = K_nearest(trainM,testM)
	# 	trainInst = K_nearest(trainM,testM)

	# 	trainR = trainInst.getVotes(k,'train')
	# 	testR = testInst.getVotes(k,'test')

	# 	trainError.append(trainR/trainRowNum)
	# 	testError.append(testR/testRowNum)

	# plt.plot(oddNumList,trainError,color='r',label="Training")
	# plt.plot(oddNumList,testError,color='b',label="Testing")

	# plt.xlabel('K-values', fontsize=16)
	# plt.ylabel('Error', fontsize=16)
	# plt.legend()
	# plt.show()





