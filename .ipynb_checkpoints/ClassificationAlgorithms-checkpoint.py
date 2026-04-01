from enum import unique
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sqlalchemy import distinct
from sympy import flatten
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from bayes import naiveBayes
import knn
import onerule
import decisionTree

# Loading the dataset
numDataset = numpy.loadtxt("seeds_dataset.txt")
labels = pd.DataFrame(data = numDataset[:, 7])
dataset = pd.DataFrame(data = numDataset[:, 0:7], columns=["area", "perimeter", "compactness", 
                                                           "length", "width", "asymmetry", "groove"])
print("The size of the dataset: " + len(dataset) + "records")
print("\n\n")

# Screening for outliers and zero values
outliers = []

for i in range(7):
    column = numDataset[:, i]
    z = numpy.abs(stats.zscore(column))
    colOutliers = numpy.where(z > 3)
    outliers.append(colOutliers)

for i in range(len(dataset)):
    flag = 0
    for j in range(7):
        if (dataset.values[i][j] == 0):
            flag = flag + 1
    if (flag > 0):
        dataset.drop(i)
        labels.drop(i)



outliers = flatten(outliers)
outliers = set(outliers)

dataset = dataset.drop(outliers)
labels = labels.drop(outliers)
            
# print(len(dataset))
# print(len(labels))

# plt.hist(labels)
# plt.show()

# Normalization (min-max)
scaledDataset = dataset.copy()
for column in dataset.columns:
    scaledDataset[column] = (scaledDataset[column] - scaledDataset[column].min()) / (scaledDataset[column].max() - scaledDataset[column].min())

# for i in range(7):
#     fig1, fig = plt.subplots(figsize=(6, 4))
#     fig.scatter(scaledDataset[scaledDataset.columns.values[i]].tolist(), labels)
#     fig.set_xlabel(scaledDataset.columns.values[i])
#     fig.set_ylabel("class")
#     plt.show()


scaledDataset = numpy.array(scaledDataset)
labels = numpy.array(labels).astype(numpy.int16)

bins = 100
trainX, testX, trainY, testY = train_test_split(
    scaledDataset, labels, test_size=0.2, random_state=42)

while (1):
    print("Please select an algorithm to run: 1 - 1-Rule, 2 - Naive Bayes, 3 - Decision Tree, 4 - k Nearest Neighbours")
    option = int(input())

    if option == 1:
        train, test = oneRule(trainX, trainY, testX, testY, bins)
        print("1-rule accuracy with {} bins: training = {}, testing = {}".format(bins, train, test))
    elif option == 2:
        train, test = naiveBayes(trainX, testX, trainY, testY, bins)
        print("Naive Bayes accuracy with {} bins: training = {}, testing = {}".format(bins, train, test))
    elif option == 3:
        tree = DecisionTree(max_depth=5)
        tree.trainTree(trainX, trainY)
        tree.evaluate(trainX, trainY, testX, testY)
    elif option == 4:
        kNearestNeighbours(trainX, testX, trainY, testY)
    elif option == 0:
        break
    else:
        print("Incorrect option")



