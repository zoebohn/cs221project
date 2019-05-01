import csv
import json
from sklearn import linear_model

senateHandleToName = "senate_handles.csv"
houseHandleToName = "house_handles.csv"
donors = ["bloomberg", "fahr", "koch", "las_vegas_sands", "nra", "paloma", "planned_parenthood", "uline"]
trainFile = "train.csv" 
testFile = "test.csv"


def learnPredictor(X, Y):
    print(X)
    classifier = linear_model.SGDClassifier()
    classifier.fit(X,Y)
    return classifier

def bagOfWordsExtractor(text):
    tweet = text["text"]
    return list(tweet.split())

def collectExamplesForPolitician(handle, X, Y, featureExtractor, hasDonor):
    try:
        with open("Tweets/" + handle + ".json") as json_file:
            data = json.load(json_file)
            for tweet in data:
                X.append(featureExtractor(tweet))
                Y.append(hasDonor)
    except IOError:
        print ("Can't find file: Tweets/" + handle + ".json")

def collectExamplesForDonor(donor, politicianFile, featureExtractor, candidateToHandleMap):
    recipients = []
    X = []
    Y = []

    with open("donors/" + donor + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            recipients.append(row)

    with open(politicianFile) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            collectExamplesForPolitician(candidateToHandleMap[row[0]], X, Y, featureExtractor, 1 if row[0] in recipients else 0)

    return (X,Y)

def buildCandidateToHandleMap():
    candidateToHandleMap = {}

    with open(houseHandleToName) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            candidateToHandleMap[row[0]] = row[1]

    with open(senateHandleToName) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            candidateToHandleMap[row[0]] = row[1]

    return candidateToHandleMap

def runDonor(donor, candidateToHandleMap):
    trainExamples = collectExamplesForDonor(donor, trainFile, bagOfWordsExtractor, candidateToHandleMap)
    classifier = learnPredictor(trainExamples[0], trainExamples[1])
    testExamples = collectExamplesForDonor(donor, testFile, bagOfWordsExtractor, candidateToHandleMap)
    classifier.score(X, Y)

candidateToHandleMap = buildCandidateToHandleMap()
for donor in donors:
    runDonor(donor, candidateToHandleMap)
