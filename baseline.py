import csv
import json

senateHandleToName = "senate_handles.csv"
houseHandleToName = "house_handles.csv"
handleToName = houseHandleToName
donors = ["bloomberg", "fahr", "koch", "las_vegas_sands", "nra", "paloma", "planned_parenthood", "uline"]
trainExamples = 
testExamples = 
numIters = 1000     # check
eta = 0.1           # check


def verbosePredict(phi, y, weights):
    yy = 1 if dotProduct(phi, weights) >= 0 else -1
    if y:
        print 'Truth: %s, Prediction: %s [%s]' % (y, yy, 'CORRECT' if y == yy else 'WRONG')
    else:
        print 'Prediction:', yy
    for f, v in sorted(phi.items(), key=lambda (f, v) : -v * weights.get(f, 0)):
        w = weights.get(f, 0)
        print "%-30s%s * %s = %s" % (f, v, w, v * w)
    return yy

def outputErrorAnalysis(testExamples, featureExtractor, weights):
    for x, y in testExamples:
        verbosePredict(featureExtractor(x), y, weights)

def dotProduct(d1, d2):
    if len(d1) < len(d2):
        return dotProduct(d2, d1) 
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def learnPredictor(trainExamples, featureExtractor):
    weights = {}  # feature => weight
    for i in range(numIters):
        for example in trainExamples:
            features = featureExtractor(example[0])
            if dotProduct(weights, features) * float(example[1]) < 1:
                for feature in features:
                    value = features[feature] 
                    if feature not in weights:
                        weights[feature] = 0.0 
                    gradient = -float(value) * example[1]
                    weights[feature] = weights[feature] - (eta * gradient)
    return weights

def bagOfWordsExtractor(text):
    results = {} 
    for word in text.split():
        if not word in results:
            results[word] = 1
        else:
            results[word] += 1
    return results

def collectExamplesForPolitician(handle, trainExamples, hasDonor):
    with open("Tweets/" + handle + ".json") as json_file:
        data = json.load(json_file)
        for tweet in data:
            trainExamples.append((tweet, hasDonor))

def collectExamplesForDonor(donor, politicianFile, candidateToHandleMap):
    recipients = []
    trainExamples = []

    with open("donors/" + donor + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            recipients.append(row)

    with open(politicianFile) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            collectExamplesForPolitician(candidateToHandleMap[row], trainExamples, row in recipients)

    return trainExamples

def buildCandidateToHandleMap():
    candidateToHandleMap = {}

    with open(handleToName) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            candidateToHandleMap[row[0]] = row[1]

    return candidateToHandleMap

def runDonor(donor, candidateToHandleMap):
    trainExamples = collectExamplesForDonor(donor, trainFile, candidateToHandleMp)
    weights = learnPredictor(trainExamples, bagOfWordsExtractor)
    testExamples = collectExamplesForDonor(donor, testFile, candidateToHandleMap)
    outputErrorAnalysis(testExamples, bagOfWordsExtractor, weights)

def main():
    candidateToHandleMap = buildCandidateToHandleMap()
    for donor in donors:
        runDonor(donor, candidateToHandleMap)
