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
DEBUG = 0

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
		self.featureM = self.data[:,1:]
		self.trueM = self.data[:,0]	
		#print len(self.featureM)



class DecisionTree():
	def __init__(self, file):
		# training data
		self.train_featureM = trainM.featureM
		self.train_trueM 	= trainM.trueM

		# testing data
		self.test_featureM 	= testM.featureM
		self.test_trueM 	= testM.trueM

		self.__row 	= len(self.train_featureM)
		self.__col 	= len(self.train_featureM[0])
		self.error	= 0


	def split(self):
		attriNum = self.__col

		for i in range(attriNum):
			














