#!/usr/bin/python

'''
Class: Oregon State University. Machine Learning - CS434
Project: Linear Regression (Assignment 1)
Author: Jiongcheng, Jiayi Du, Tao Chen
'''

import sys
import csv
import time
import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

TRAINING_DATA = 'data/housing_train.txt'
TESTING_DATA = 'data/housing_test.txt'

class Matrix:
	def __init__(self, file, dummy):
		self.data = self.__read_data(file)
		self.row = len(self.data)
		self.col = len(self.data[1])
		self.__addDummyVar = dummy
		self.__getMatrix()

	def __read_data(self, file):
		with open(file) as textFile:
			lines = [line.split() for line in textFile]
			return lines

	def __getMatrix(self):
		if self.__addDummyVar:
			self.xM = [[1 for x in range(self.col)] for y in range(self.row)] 
			self.yM = [[0 for x in range(1)] for y in range(self.row)] 	

			for i in range(self.row):
				for j in range(self.col-1):
					self.xM[i][j+1] = float(self.data[i][j])

			for i in range(self.row):
				self.yM[i][0] = float(self.data[i][self.col-1])

		# NOT adding dummy data
		else:
			self.xM = [[0 for x in range(self.col-1)] for y in range(self.row)] 
			self.yM = [[0 for x in range(1)] for y in range(self.row)] 	

			for i in range(self.row):
				for j in range(self.col-1):
					self.xM[i][j] = float(self.data[i][j])

			for i in range(self.row):
				self.yM[i][0] = float(self.data[i][self.col-1])

		

class Algorithm:
	def __init__(self,trainM,testM,featureNum,lammda):
		self.train_M = trainM
		self.test_M = testM

		self.__addFeatureCol(featureNum)
		self.__lammda = lammda

	def __addFeatureCol(self, n):
		for i in range(n):
			a = random.randint(1,20)	# 20 indicates 
			for row1 in self.test_M.xM:	# row num same in train and test
				row1.append(round(random.uniform(0,a),4))

			for row2 in self.train_M.xM:
				row2.append(round(random.uniform(0,a),4))

	def getWeightVec(self):
		x = np.array(self.train_M.xM)
		y = np.array(self.train_M.yM)
		xt = x.transpose()
		id_M = np.identity(len(xt))
		lamda_IP = id_M * self.__lammda
		invP = inv(np.dot(xt,x)+lamda_IP)
		xtyP = np.dot(xt,y)
		weightVector = np.dot(invP,xtyP)
		return weightVector

	def __getSSE(self,m):
		w = self.getWeightVec()
		y = np.array(m.yM)	
		x = np.array(m.xM)	
		p = y - np.dot(x,w)
		SSE = np.dot(p.transpose(),p)
		return SSE[0][0]

	def get_train_SSE(self):
		return self.__getSSE(self.train_M)

	def get_test_SSE(self):
		return self.__getSSE(self.test_M)




class Prog:
	def __init__(self):
		self.__traning_M = Matrix(TRAINING_DATA, True)
		self.__testing_M = Matrix(TESTING_DATA, True)
		self.__traning_M_noDummy = Matrix(TRAINING_DATA, False)
		self.__testing_M_noDummy = Matrix(TESTING_DATA, False)
	
	def question_1to4(self):
		result = Algorithm(self.__traning_M,self.__testing_M,0,0)
		result_noDummy = Algorithm(self.__traning_M_noDummy,self.__testing_M_noDummy,0,0)

		train_result = result.get_train_SSE()
		train_result_noDummy = result_noDummy.get_train_SSE()

		test_result = result.get_test_SSE()
		test_result_noDummy = result_noDummy.get_test_SSE()

		print "Training (with Dummy):",train_result
		print "Training (No Dummy):",train_result_noDummy
		print "Testing (with Dummy):",test_result
		print "Testing (no Dummy):",test_result_noDummy
		print "\n"


	def question_5(self):
		FEATURE_COL = 200 	# 0 ~ 25
		train_y = []
		test_y = []
		train_x = test_x = np.arange(0, FEATURE_COL)

		for i in range(FEATURE_COL):
			__traning_M = Matrix(TRAINING_DATA, True)
			__testing_M = Matrix(TESTING_DATA, True)
			result = Algorithm(__traning_M,__testing_M,i,0)
			train_y.append(result.get_train_SSE())
			test_y.append(result.get_test_SSE())

		plt.plot(train_x,train_y,'r', label='Training Data')
		plt.plot(test_x,test_y,'b', label='Testing Data')
		plt.xlabel('Added Columns (Features)', fontsize=16)
		plt.ylabel('SSE', fontsize=16)
		plt.title('Question 5')
		plt.legend()
		plt.show()

	def question_6(self):
		train_y = []
		test_y = []
		xAxie = []

		a = np.linspace(0.0001,100,100,endpoint=False)
		b = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 15, 20, 25, 30]

		for l in a:
			xAxie.append(l)
			__traning_M = Matrix(TRAINING_DATA, True)
			__testing_M = Matrix(TESTING_DATA, True)
			result = Algorithm(__traning_M,__testing_M,0,l)
			train_y.append( result.get_train_SSE() )
			test_y.append( result.get_test_SSE() )

		plt.subplot(1,2,1)
		plt.plot(xAxie,train_y,color='b')
		plt.title("Training Data")
		plt.xlabel('Lamda(0.0001 ~ 100)', fontsize=16)
		plt.ylabel('SSE', fontsize=16)

		plt.subplot(1,2,2)
		plt.plot(xAxie,test_y,color='r')
		plt.title("Testing Data")
		plt.xlabel('Lamda(0.0001 ~ 100)', fontsize=16)
		plt.ylabel('SSE', fontsize=16)
		plt.show()

	def question_7to8(self):
		lam = np.linspace(0.0001,100,100,endpoint=False)
		xAxie = []
		weight_y = []
		w_vec_y = []
		for l in lam:
			xAxie.append(l)
			__traning_M = Matrix(TRAINING_DATA, True)
			__testing_M = Matrix(TESTING_DATA, True)
			result = Algorithm(__traning_M,__testing_M,0,l)
			w_vec_y.append(result.getWeightVec()[0][0])
			weightNorm = np.linalg.norm(result.getWeightVec(), ord=2)
			weight_y.append(weightNorm)

		plt.plot(xAxie,weight_y,color='g')
		plt.xlabel('Lamda(0.0001 ~ 100)', fontsize=16)
		plt.ylabel('Norm of Weight', fontsize=16)
		plt.title("Norm of Weight Vector vs. Lamda")
		plt.show()



#-----------------------
#		Main
#-----------------------

if __name__ == "__main__":
	prog = Prog()
	argc = len(sys.argv)

	if (argc == 2):
		q = sys.argv[1]
		if q == '1':
			print "---------- Question 1 to 4 ----------"
			prog.question_1to4()
		elif q == '2':
			print "---------- Question 5 ----------"
			print "Loading graph..."
			prog.question_5()
		elif q == '3':
			print "---------- Question 6 ----------"
			print "Loading graph..."
			prog.question_6()
		elif q == '4':
			print "---------- Question 7 to 8 ----------"
			print "Loading graph..."
			prog.question_7to8()
		else:
			print '[USAGE]: python program.py 1'
	else:
		print '[USAGE]: python program.py 1'











