import numpy

def naiveBayes(trainDataset, testDataset, labels, testLabels, bins):
    classesP, featuresP = trainBayes(trainDataset, labels, bins)

    accuracyTrain, _ = testBayes(trainDataset, labels, bins, classesP, featuresP)
    accuracyTest, cmTest = testBayes(testDataset, testLabels, bins, classesP, featuresP)

    return accuracyTrain, accuracyTest, cmTest


def trainBayes(dataset, labels, bins):
    classesP = numpy.zeros((3))
    for l in labels:
        classesP[l[0] - 1] += 1

    classesP = classesP / len(labels)

    featuresP = numpy.zeros((3, 7, bins + 1))
    for i in range(len(dataset)):
        for j in range(7):
            valueIndex = min(int(dataset[i][j] * bins), bins)
            featuresP[labels[i][0] - 1][j][valueIndex] += 1

    featuresP += 1
    for i in range(3):
        featuresP[i] /= (classesP[i] * len(labels) + bins + 1)

    return classesP, featuresP


def testBayes(dataset, labels, bins, classesP, featuresP):
    correct = 0
    confusion_test = numpy.zeros((3, 3))

    for i in range(len(dataset)):
        probabilities = []

        for j in range(len(classesP)):
            classProbability = numpy.log(classesP[j])

            for f in range(7):
                valueIndex = min(int(dataset[i][f] * bins), bins)
                classProbability += numpy.log(featuresP[j][f][valueIndex] + 1e-9)

            probabilities.append(classProbability)

        finalClass = numpy.argmax(probabilities) + 1
        trueClass = labels[i][0]

        confusion_test[int(trueClass) - 1][int(finalClass) - 1] += 1

        if finalClass == trueClass:
            correct += 1

    return correct / len(dataset), confusion_test