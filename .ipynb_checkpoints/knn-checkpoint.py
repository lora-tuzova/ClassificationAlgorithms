import numpy as np
from collections import Counter

def kNearestNeighbours(datasetTrain, datasetTest, labelsTrain, labelsTest):
    
    best_accuracy = -1
    best_k = None
    best_confusion = None

    for k in range(1, 11):
        accuracy = 0
        confusion = np.zeros((3, 3))

        for point in range(len(datasetTest)):
            distances = []
            indexes = []

            for i in range(len(datasetTrain)):
                distances.append(euclids(datasetTest[point], datasetTrain[i]))
                indexes.append(i)

            distances, indexes = zip(*sorted(zip(distances, indexes)))
            neighbours = indexes[:k]

            classes = []
            for n in neighbours:
                classes.append(labelsTrain[n][0])

            finalClass = Counter(classes).most_common(1)[0][0]
            trueClass = labelsTest[point][0]

            confusion[int(trueClass) - 1][int(finalClass) - 1] += 1

            if finalClass == trueClass:
                accuracy += 1

        accuracy /= len(datasetTest)

        print("Accuracy of {} neighbours: {}".format(k, accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_confusion = confusion

    return best_k, best_confusion


def euclids(x, y):
    if x.shape == y.shape:
        distance = 0
        for dim in range(x.shape[0]):
            if dim == 0 or dim == 1:
                distance += 4 * (x[dim] - y[dim]) ** 2
            elif dim == 5 or dim == 6:
                distance += 2 * (x[dim] - y[dim]) ** 2
            else:
                distance += (x[dim] - y[dim]) ** 2

        return np.sqrt(distance)
    else:
        return 0