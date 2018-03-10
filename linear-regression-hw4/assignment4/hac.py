#!/usr/bin/python

'''
Class: Oregon State University. Machine Learning - CS434
Project: Linear Regression (Assignment 4)
Author: Jiongcheng, Jiayi Du, Tao Chen
'''

from __future__ import division

import math
import random
import numpy as np
import csv
import matplotlib.pyplot as plt

class Data:
    def __init__(self, fileName):
        self.__readFile(fileName)

    def __readFile(self, fileName):
        openFile = open(fileName, 'r')
        readCSV = csv.reader(openFile, delimiter=',')
        self.__rawData = []

        for row in readCSV:
            self.__rawData.append(map(float, row))

        # mark each point as an individual cluster
        self.__currentClusters = []
        i = 0
        for point in self.__rawData:
            self.__currentClusters.append([])
            self.__currentClusters[i].append(list(point))
            i = i + 1

    # find the two clusters to merge
    def closestClusters(self, ifPrint):
        tempClusters = list(self.__currentClusters)
        size = len(tempClusters)
        pairs = []
        for i in range(size):
            j = i + 1
            for j in range(i+1, size):
                # if method == "single":
                distance = self.__singleLink(tempClusters[i], tempClusters[j])
                # if method == "complete":
                    # distance = self.__completeLink(tempClusters[i], tempClusters[j])
                pairs.append([i, j, distance])
        pairs.sort(key=lambda x: x[2])
        #print len(pairs)
        if ifPrint == 0:
            for pair in pairs:
                print pair
        self.__mergeClusters(pairs[0][0], pairs[0][1])

    # merge the two clusters that are connected by the shortest link
    def __mergeClusters(self, a, b):
        cluster1 = list(self.__currentClusters[a])
        cluster2 = list(self.__currentClusters[b])

        newCluster = []
        for point in cluster1:
            newCluster.append(point)
        for point in cluster2:
            newCluster.append(point)
        #print newCluster
        self.__currentClusters.append(list(newCluster))

        self.__currentClusters.remove(cluster1)
        self.__currentClusters.remove(cluster2)

        #print self.__currentClusters

    # calculate the distance using the singe link method
    def __singleLink(self, cluster1, cluster2):
        distances = []
        for point1 in cluster1:
            for point2 in cluster2:
                distances.append(self.__eucliDist(point1, point2))

        distances.sort()
        return distances[0]

    # calculate the distance using the complete link method
    def __completeLink(self, cluster1, cluster2):
        distances = []
        for point1 in cluster1:
            for point2 in cluster2:
                distances.append(self.__eucliDist(point1, point2))

        distances.sort().reverse()
        return distances[0]

    # calculate the euclidean distance
    def __eucliDist(self, point1, point2):
        distSum = sum([(p1-p2)**2 for p1, p2 in zip(point1, point2)])
        return math.sqrt(distSum)

    # return the current clusters
    def getClusters(self):
        return self.__currentClusters

if __name__ == "__main__":
    trainingSet = Data("data/data-2.txt")
    while(len(trainingSet.getClusters()) > 10):
        trainingSet.closestClusters(1)
    print "=============================="
    while(len(trainingSet.getClusters()) > 1):
        trainingSet.closestClusters(0)
    print "=============================="
    print len(trainingSet.getClusters())
