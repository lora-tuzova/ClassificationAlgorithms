import numpy
from collections import Counter
from functools import reduce


def kNearestNeighbours(dataset, labels):
    
    for k in range(1, 11):
        accuracy = 0
        for point in range(len(dataset)):
            distances = []
            indexes = []
            for i in range (len(dataset)):
                if i != point:
                    distances.append(euclids(dataset[point], dataset[i]))
                    indexes.append(i)
            distances, indexes = zip(*sorted(zip(distances, indexes)))
            neighbours = indexes[:k]
            classes = []
            for n in neighbours:
                classes.append(labels[n][0])
            
            sums = []

            for c in set(classes):
                sumC = 0
                for n in range(len(neighbours)):
                    if classes[n] == c:
                        sumC += 1 / (distances[n] * distances[n])
                sums.append([c, sumC])

            # finalClass = reduce(lambda x, y: x if x[1] > y[1] else y, sums)
            # if finalClass[0] == labels[point][0]:
            #     precision += 1

            finalClass = Counter(classes).most_common(1)[0][0]
            if finalClass == labels[point][0]:
                accuracy += 1

        accuracy /= len(dataset)
        print ("Accuracy of {} neighbours: {}".format(k, accuracy))

    return()

def euclids(x, y):
    if x.shape == y.shape:
        distance = 0
        for dim in range(x.shape[0]):
            # if dim == 0 or dim == 1:
            #     distance += 4 * (x[dim] - y[dim]) * (x[dim] - y[dim])
            # else:
            #     if dim == 5 or dim == 6:
            #         distance += 2 * (x[dim] - y[dim]) * (x[dim] - y[dim])
            #     else:
                      distance += (x[dim] - y[dim]) * (x[dim] - y[dim])

        return numpy.sqrt(distance)
    else:
        return 0