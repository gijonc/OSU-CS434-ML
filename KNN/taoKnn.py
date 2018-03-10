from __future__ import division
# CS 434
# Assignment 3
# Tao Chen

import math
import numpy as np
import csv
import matplotlib.pyplot as plt

class Data:
    def __init__(self, fileName):
        self.__readFile(fileName)

    def __readFile(self, fileName):
        openFile = open(fileName, 'r')
        readCSV = csv.reader(openFile, delimiter = ',')
        self.__rawData = []

        for row in readCSV:
            self.__rawData.append(map(float, row))

        self.normalizeData()


    def normalizeData(self):
        maxOfCol = []
        minOfCol = []

        rowN = len(self.__rawData)
        colN = len(self.__rawData[0])

        # axis = 0 count for col of the array
        maxOfCol = np.max(self.__rawData,axis=0)
        minOfCol = np.min(self.__rawData,axis=0)

        for i in range(rowN):
            for j in range(colN): # pass the first column
                num = self.__rawData[i][j]
                minV = minOfCol[j]
                maxV = maxOfCol[j]
                self.__rawData[i][j] = (num - minV)/(maxV - minV)


    def getRawData(self):
        return self.__rawData

    # Use the training data set to classify testing data with K
    def test(self, trainData, K):
        correct = 0
        incorrect = 0
        for rowTest in  self.__rawData:
            # find the first K elements in the sorted list
            distances = self.eucliDistList(trainData, rowTest[1:len(rowTest)])
            for row in distances[0:K]:
                benign = 0
                malignant = 0
                # the second element in each list is the lable
                if row[1] == -1:
                    benign += 1
                else:
                    malignant += 1

            if benign > malignant and rowTest[0] == -1:
                correct += 1
            elif benign < malignant and rowTest[0] == 1:
                correct += 1
            else:
                incorrect += 1
        # ================================================
        return incorrect/(correct + incorrect)

    # leave-one-out cross-validation
    def validationTesting(self, K):
        rows = 1
        correct = 0
        incorrect = 0
        length = len(self.__rawData)
        for leaveOutRow in self.__rawData:
            # remove that row from the data
            newTrainData = list(self.__rawData)[0:rows] + list(self.__rawData)[rows:length]
            # validate using the leave-out row
            distances = self.eucliDistList(newTrainData, leaveOutRow[1:len(leaveOutRow)])
            for row in distances[0:K]:
                benign = 0
                malignant = 0
                # the second element in each list is the lable
                if row[1] == -1:
                    benign += 1
                else:
                    malignant += 1

            if benign > malignant and leaveOutRow[0] == -1:
                correct += 1
            elif benign < malignant and leaveOutRow[0] == 1:
                correct += 1
            else:
                incorrect += 1

            rows += 1

        return incorrect/(correct + incorrect)

    # returns the sorted list of euclidean distance
    def eucliDistList(self, trainData, testFeatures):
        distances = []
        for rowTrain in trainData:
            trainFeatures = rowTrain[1:len(rowTrain)]
            # rowTrain and rowTest are list objects
            eucliDist = math.sqrt(sum([(test - train)**2 for test, train in zip(testFeatures, trainFeatures)]))
            newRowWithDist = rowTrain[0:len(rowTrain)]
            newRowWithDist.insert(0, eucliDist)
            distances.append(newRowWithDist)
        # sort the array by the euclidean distance, which would be the first element
        distances.sort(key=lambda x: x[0])
        return distances

if __name__ == "__main__":
    train = Data("data/knn_train.csv")
    test = Data("data/knn_test.csv")

    Ks = []
    errorsFromTesting = []
    errorsFromSelfTest = []
    errorsFromValidation = []


    #train.normalizeData()

    for i in range(30):
        Ks.append(2 * i + 1)
    for K in Ks:
        errorsFromTesting.append(test.test(train.getRawData(), K))
        errorsFromSelfTest.append(train.test(train.getRawData(), K))
        errorsFromValidation.append(train.validationTesting(K))
    #print errorsFromSelfTest
    #print errorsFromSelfTest

    plt.plot(Ks, errorsFromTesting, color="r", label="Testing Error")
    plt.plot(Ks, errorsFromSelfTest, color="b", label="Training Error")
    plt.plot(Ks, errorsFromValidation, color="y", label="Validation Error")
    plt.title("Error Rate vs. K-Values")
    plt.xlabel('K-values', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.legend()
    plt.show()
