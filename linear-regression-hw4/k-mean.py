#!/usr/bin/python

'''
Class: Oregon State University. Machine Learning - CS434
Project: Linear Regression (Assignment 4)
Author: Jiongcheng, Jiayi Du, Tao Chen
'''

import sys
import csv
import math
import time
import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

TRAINING_DATA = 'data/data-1.txt'
# TRAINING_DATA = 'data-1.txt'
DEBUG = 1


class K_mean:
	def __init__(self,file,k):
		self.k = k
		self.data = self.__read_data(file)
		self.rowN = len(self.data)
		self.colN = len(self.data[1])

	def __read_data(self, file):
		with open(file, 'r') as f:
			data = [row for row in csv.reader(f.read().splitlines())]
    		return np.array(data).astype(np.int)

	def __euclideanDist(self,a,b):
		# a and b are as a list (row)
		# len(a) should = len(b)
		if len(a) == len(b):
	 		d = 0.0
	 		for i in range(len(a)):
	 			d += pow( (float(a[i]) - float(b[i])) , 2)
	 		return math.sqrt(d)
	 	else:
	 		print "[Euclidean:] size not matched"


	def getInitCentroid(self):	
		centroidList = []
		c = []
		# get k number of initial cluster centers
		# Select first centroid randomly from dataset
		# Select next initial centroid that the Euclidean distance of that object 
		# is maximum from other selected initial centroids

		for _ in range(self.k):
			r = random.randint(0,self.rowN-1)	# random number from row#
			centroidList.append( self.data[r] )	# random point 

		return centroidList	# this returns the row number of the random point of the dataset


	def updateCluster(self, centroid = []):
		cluster = []

		for i in range(self.k):
			cluster.append([])	# create k x n array

		for cur_instance in self.data:
			distSet = []

			# Assign data point to the Centroid whose distance from the Centroid is minimum
			for c in centroid:
				dist = self.__euclideanDist( cur_instance, c )
				distSet.append( dist )
				#print c,":",dist
			
			centroid_Idx = distSet.index( min(distSet) )
			cluster[centroid_Idx].append(cur_instance)

		return cluster


	def updateCentroid(self,cluster = []):
		centroidList = []

		for i in range(len(cluster)):
			m = np.array(cluster[i])
			centroidList.append( m.mean(axis=0) )	# mean of all columns

		return centroidList


	def getSSE(self,cluster = [],centroid = []):
		sse = 0.0
		if len(cluster) == len(centroid) == self.k:

			for i in range( len(cluster) ):
				for point in cluster[i]:
					sse += pow(self.__euclideanDist(point,centroid[i]),2)
			return sse



	def run(self):
		d_sse = 9999999999
		prev_sse = 0
		iterate = 0

		sse = []
		# create k clusters
		init_centroids = self.getInitCentroid()
		cluster = self.updateCluster(init_centroids)

		for iterate in range(i):
		# while d_sse != 0:	# stop by convergence
			centroid = self.updateCentroid(cluster)
			_sse = self.getSSE(cluster,centroid)


			print "---------------",iterate+1,"---------------"
			# self.printClusterInfo(cluster)
			print "SSE:", _sse
			sse.append(_sse)

			# iterate += 1
			d_sse = _sse - prev_sse
			print "d_SSE:", d_sse

			prev_sse = _sse
			# update (re-assign) cluster
			cluster = self.updateCluster(centroid)

		return sse



	def run_random(self):
		d_sse = 9999999999
		prev_sse = 0
		iterate = 0

		sse = []
		# create k clusters
		init_centroids = self.getInitCentroid()
		# print "Initial centroids:",init_centroids
		cluster = self.updateCluster(init_centroids)

		while d_sse != 0:	# stop by convergence
			centroid = self.updateCentroid(cluster)
			_sse = self.getSSE(cluster,centroid)

			print "---------------",iterate+1,"---------------"
			self.printClusterInfo(cluster)
			print "SSE:", _sse
			sse.append(_sse)

			iterate += 1
			d_sse = _sse - prev_sse
			print "d_SSE:", d_sse
			prev_sse = _sse


			# update (re-assign) cluster
			cluster = self.updateCluster(centroid)
		return sse


	def printClusterInfo(self,cluster = []):
		print "# of cluster : # of point" 
		for i in range(self.k):
			print i,":",len(cluster[i])




# ---------------- Global Function ---------------------

def multipleRun():
	K = range(2,11)	# k = 2~10
	minSSE = []
	sseList = []


	for k in K:
		minSSE[:] = []	# clean the list

		data = K_mean(TRAINING_DATA,k)
		print "------- k=",k,"--------"
		for _ in range(1):	# iterate 10 times for each k-value
			sse = data.run_random()	
			print sse		
			minSSE.append(round(sse,1))
		m = min(minSSE)
		print "min:",m
		sseList.append(m)

	print "------- Result --------"
	print "length:",len(sseList)
	print sseList

	plt.title("K-Means")
	plt.xlabel('K-value', fontsize=16)
	plt.ylabel('SSE', fontsize=16)
	plt.plot(K,sseList)
	plt.show()



if __name__ == "__main__":

# first question
	# data = K_mean(TRAINING_DATA,2)
	# ySSE = data.run_random()
	# xList = range(0,len(ySSE))

	# plt.title("K-Means")
	# plt.xlabel('Iteration', fontsize=16)
	# plt.ylabel('SSE', fontsize=16)
	# plt.plot(xList,ySSE)
	# plt.show()

# second question
	# multipleRun()

	K = range(2,11)	# k = 2~10

	# recorded sse from multiple run
	ySSE = [16896104221.1, 15854617698.5, 15080072856.4, 
			14605120078.6, 14187745996.1, 13882013790.2,
			13440695985.7, 13409677064.8, 12940374008.7]
	plt.title("K-Means")
	plt.xlabel('K-value', fontsize=16)
	plt.ylabel('SSE', fontsize=16)
	plt.plot(K,ySSE)
	plt.show()
