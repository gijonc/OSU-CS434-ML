from __future__ import division
# CS 434
# Assignment 3 Desicion trainFeatures
# Tao Chen

import math
import numpy as np
import csv
import matplotlib.pyplot as plt

def calEntropy(dataSet):
    benign = 0
    malignant = 0
    # the fisrt column is the labels
    for row in dataSet:
        if row[0] == -1:
            benign += 1
        else:
            malignant += 1

    total = benign + malignant
    benignPr = benign / total
    malignantPr = malignant / total

    # to prevent math domain error
    if benignPr == 0 or malignantPr == 0:
        return 0

    # benign is negative, malignant is positive
    entropy = -benign * math.log(benignPr) - malignantPr * math.log(malignantPr)

    return entropy

class Data:
    def __init__(self, fileName):
        self.__readFile(fileName)

    def __readFile(self, fileName):
        openFile = open(fileName, 'r')
        readCSV = csv.reader(openFile, delimiter = ',')
        self.__rawData = []

        for row in readCSV:
            self.__rawData.append(map(float, row))

    def getRawData(self):
        return self.__rawData

    def findSplit(self, dataSet):
        # make a copy of the data
        parentSet = list(dataSet)
        parentEntropy = calEntropy(parentSet)
        # try every attribute
        numOfAttr = len(self.__rawData[0]) - 1
        numOfPoints = len(parentSet)
        highestGain = -1
        splitThreshold = 0
        splitAttr = 0
        for i in range(numOfAttr):
            # sort the points with respect to the attribute
            parentSet.sort(key=lambda x: x[i+1])
            # go from the smallest one to the largest one
            for k in range(numOfPoints-1):
                threshold = parentSet[k][i+1]
                # calculate the information gain
                # binary construction
                subset_1 = parentSet[0:k+1]
                subset_2 = parentSet[k+1:numOfPoints]
                # To keep the lines short
                sumOfSubsetEntropy = len(subset_1)/numOfPoints * calEntropy(subset_1)
                sumOfSubsetEntropy += len(subset_2)/numOfPoints * calEntropy(subset_2)
                gain = parentEntropy - sumOfSubsetEntropy
                # record the biggest information gain
                if gain > highestGain:
                    highestGain = gain
                    splitThreshold = threshold
                    splitAttr = i + 1

                    malignant = 0
                    benign = 0
                    # for row in subset_1:
                    #     if row[0] == -1:
                    #         benign += 1
                    #     else:
                    #         malignant += 1
                    # print "subset_1:" , benign / (benign + malignant), len(subset_1)
                    # malignant = 0
                    # benign = 0
                    # for row in subset_2:
                    #     if row[0] == -1:
                    #         benign += 1
                    #     else:
                    #         malignant += 1
                    # print "subset_2:" , benign / (benign + malignant), len(subset_2)

        return (splitAttr, splitThreshold, highestGain)

    def classify(self, splitAttr, splitThreshold):
        correct = 0
        incorrect = 0
        for row in self.__rawData:
            # if it's smaller, it's benign, otherwise malignant
            if row[splitAttr] <= splitThreshold and row[0] == -1:
                correct += 1
            elif row[splitAttr] > splitThreshold and row[0] == 1:
                correct += 1
            else:
                incorrect += 1
        print correct / (correct + incorrect), correct, incorrect


if __name__ == "__main__":
    train = Data("data/knn_train.csv")
    test = Data("data/knn_test.csv")
    (splitAttr, splitThreshold, highestGain) = train.findSplit(train.getRawData())
    print "Attribute, Threshold, Highest Gain"
    print splitAttr, splitThreshold, highestGain
    # apply the learned stump on the training data
    print "Percentage, Number of Correct, Number of Incorrect"
    train.classify(splitAttr, splitThreshold)
    test.classify(splitAttr, splitThreshold)
