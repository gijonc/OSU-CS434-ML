
import sys
import csv
import time
import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math

TRAINING_DATA = 'data/usps-4-9-train.csv'
TESTING_DATA = 'data/usps-4-9-test.csv'

THETA = 0.001

#ETA  = 5e-7
ETA  = 5e-9
#ETA  = 5e-11

class Matrix:
	def __init__(self,file):
		self.data = self.__read_data(file)
		self.row = len(self.data)
		self.col = len(self.data[0])
		self.__addDummyVar = True
		self.__getMatrix()

	def __read_data(self,file):
		data = np.genfromtxt(file, delimiter=',')
		return data
		return data



	def __getMatrix(self):
		if self.__addDummyVar:
			self.xM = [[1 for x in range(self.col)] for y in range(self.row)] 
			self.yM = [[0 for x in range(1)] for y in range(self.row)] 	

			for i in range(self.row):
				for j in range(self.col-1):
					self.xM[i][j+1] = int(self.data[i][j])
				self.yM[i][0] = int(self.data[i][self.col-1])

		else:
			self.xM = [[0 for x in range(self.col-1)] for y in range(self.row)] 
			self.yM = [[0 for x in range(1)] for y in range(self.row)] 	

			for i in range(self.row):
				for j in range(self.col-1):
					self.xM[i][j] = int(self.data[i][j])
				self.yM[i][0] = int(self.data[i][self.col-1])



class Algorithm:
	def __init__(self,trainM,testM):
		self.trainM = trainM
		self.testM = testM
		self.__row = len(trainM.xM)
		self.__col = len(trainM.xM[0])
		self.w = np.zeros((1,self.__col))

	def __sigmoid(self,x):	return 1/(1+np.exp(-x))

	# quesiotn 1
	def batch_Learning(self):
		LossY = []
		d_normY = []
		d_prev = np.zeros((1,self.__col))
		iterate = []
		train_accuracyY = []
		test_accuracyY = []
		cnt = 0
		#while 1:	# iterate for w
		for i in range(1000):
			train_correctness = 0
			test_correctness = 0

			d = np.zeros((1,self.__col))
			l = 0
			for j in range(self.__row):
				x 	= self.trainM.xM[j]
				sigm 	= self.__sigmoid(np.dot(self.w,x))
				yi  = self.trainM.yM[j][0]
				err = yi - sigm
				d 	+= err * x

				# -----------------------------------
				#			problem 1
				#------------------------------------
				# avoid over minimum float number for log:
				if yi == 1:
					y = sigm
					if y <= sys.float_info.min:
						y = sys.float_info.min
				elif yi == 0:
					y = (1-sigm)
					if y <= sys.float_info.min:
						y = sys.float_info.min
				l += -math.log(y) 

				# -----------------------------------
				#			problem 2
				#------------------------------------
				if (sigm >= 0.5 and yi == 1) or (sigm < 0.5 and yi == 0):
					train_correctness += 1

					# testing data
			for j in range(len(self.testM.xM)):
				testX = self.testM.xM[j]
				test_sigm = self.__sigmoid(np.dot(self.w,testX))
				testy = self.testM.yM[j][0]
				if (test_sigm >= 0.5 and testy == 1) or (test_sigm < 0.5 and testy == 0):
					test_correctness += 1

			
			d_norm = np.linalg.norm(d_prev - d, ord=2)
			d_normY.append(d_norm)
			LossY.append(l)
			train_accuracyY.append(float(train_correctness)/self.__row)
			test_accuracyY.append(float(test_correctness)/len(self.testM.xM))

			# update w 
			if d_norm < THETA: # see if converge
			 	break
			else:	
				self.w += ETA * d
				d_prev = d

				iterate.append(cnt)
				cnt += 1


		print "w:",np.linalg.norm(self.w, ord=2)
		print "iterate:",len(iterate)
		print "d_norm:", d_norm

		#plt.subplot(2,2,1)
		# plt.plot(iterate,LossY,color='r')
		# plt.title('Leanring Weight = 5*10e-11')
		# plt.xlabel('Iteration', fontsize=14)
		# plt.ylabel('Loss Function', fontsize=14)

		plt.plot(iterate,train_accuracyY,color='r')
		plt.plot(iterate,test_accuracyY,color='b')

		plt.xlabel('Iteration Count', fontsize=16)
		plt.ylabel('Training Accuracy', fontsize=16)
		plt.show()




if __name__ == "__main__":
	trainM = Matrix(TRAINING_DATA)
	testM = Matrix(TESTING_DATA)

	algo = Algorithm(trainM,testM)
	algo.batch_Learning()

