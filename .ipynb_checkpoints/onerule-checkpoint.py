import numpy as np

def oneRule(trainDataset, labels, testDataset, testLabels, bins):
    # training
    featureClasses = np.zeros((7, bins+1, 3))
    rules = np.zeros((7, bins + 1))

    for i in range(len(trainDataset)):
        for j in range(7):
            featureClasses[j][min(int(trainDataset[i][j] * bins), bins)][labels[i][0] - 1] += 1

    for i in range(7):
        for j in range (bins + 1):
            rules[i][j] = np.argmax(featureClasses[i][j]) + 1

    errors = np.zeros(7)
    for i in range(len(trainDataset)):
        for j in range(7):
            if labels[i][0] != rules[j][min(int(trainDataset[i][j]*bins), bins)]:
                errors[j] += 1

    bestFeature = np.argmin(errors)
    bestRule = rules[bestFeature]

    correct = 0

    for i in range(len(trainDataset)):
        value = trainDataset[i][bestFeature]
        valueIndex = min(int(value * bins), bins)
        predicted = bestRule[valueIndex]

        if labels[i][0] == predicted:
            correct += 1

    accuracyTrain = correct / len(trainDataset)
    
    # testing
    correct_test = 0
    confusion_test = np.zeros((3, 3))

    for i in range(len(testDataset)):
        value = testDataset[i][bestFeature]
        true = testLabels[i][0]

        valueIndex = min(int(value * bins), bins)
        predicted = bestRule[valueIndex]

        # confusion matrix update
        confusion_test[int(true)-1][int(predicted)-1] += 1

        if true == predicted:
            correct_test += 1

    accuracyTest = correct_test / len(testDataset)

    return (accuracyTrain, accuracyTest, confusion_test)