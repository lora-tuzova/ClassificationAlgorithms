import numpy

def naiveBayes(trainDataset, testDataset, labels, testLabels, bins):
    classesP, featuresP = train(trainDataset, labels, bins)

    accuracy = test(testDataset, testLabels, bins, classesP, featuresP)

    return(accuracy)

    

def train(dataset, labels, bins):
    # Computing probability of each class
    classesP = numpy.zeros((3))
    for l in labels:
        classesP[l[0]-1] = classesP[l[0]-1] + 1

    classesP = classesP / len(labels)

    
    # Computing each feature value variant's probability
    featuresP = numpy.zeros((3, 7, bins+1))
    for i in range(len(dataset)):
        for j in range(7):
            valueIndex = min(int(dataset[i][j] * bins), bins)
            featuresP[labels[i][0]-1][j][valueIndex] = featuresP[labels[i][0]-1][j][valueIndex] + 1
    
    featuresP += 1
    for i in range (3):
        featuresP[i] /= (classesP[i] * len(labels) + bins + 1)
    # 3 pages with classes with 7 rows of features with columns of values

    return(classesP, featuresP)


def test(dataset, labels, bins, classesP, featuresP):
    correct = 0

    for i in range(len(dataset)): # Probabilities for each test string
        probabilities = []

        for j in range(len(classesP)): # for each possible class
            classProbability = numpy.log(classesP[j])

            for f in range(7): # for each feature in a string
                valueIndex = min(int(dataset[i][f] * bins), bins)
                classProbability += numpy.log(featuresP[j][f][valueIndex])

            probabilities.append(classProbability)

        finalClass = numpy.argmax(probabilities) + 1

        if finalClass == labels[i][0]:
            correct += 1

    return correct / len(dataset)