import numpy

def oneRule(trainDataset, labels, testDataset, testLabels, bins):
    # training
    featureClasses = numpy.zeros((7, bins+1, 3))
    rules = numpy.zeros((7, bins + 1))

    for i in range(len(trainDataset)):
        for j in range(7):
            featureClasses[j][min(int(trainDataset[i][j] * bins), bins)][labels[i][0] - 1] += 1

    for i in range(7):
        for j in range (bins + 1):
            rules[i][j] = numpy.argmax(featureClasses[i][j]) + 1

    errors = numpy.zeros(7)
    for i in range(len(trainDataset)):
        for j in range(7):
            if labels[i][0] != rules[j][min(int(trainDataset[i][j]*bins), bins)]:
                errors[j] += 1

    bestFeature = numpy.argmin(errors)
    bestRule = rules[bestFeature]

    
    # testing
    accuracy = 0

    for test in range(len(testDataset)):
        value = testDataset[test][bestFeature]
        classCorrect = testLabels[test][0]

        valueIndex = min(int(value * bins), bins)
        predicted = bestRule[valueIndex]

        if classCorrect == predicted:
            accuracy += 1

    return(accuracy / len(testLabels))
