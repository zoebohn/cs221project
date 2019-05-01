import csv
import json
from sklearn import linear_model

senateHandleToName = "senate_handles.csv"
houseHandleToName = "house_handles.csv"
donors = ["bloomberg", "fahr", "koch", "las_vegas_sands", "nra", "paloma", "planned_parenthood", "uline"]
trainFile = "train.csv" 
testFile = "test.csv"
allWords = {}

def getX(featureList):
    X = []
    for features in featureList:
        tmp = []
        for word in allWords:
            tmp.append(1 if word in features else 0)
        X.append(tmp)
    return X

def learnPredictor(X, Y):
    print(X)
    classifier = linear_model.SGDClassifier()
    classifier.fit(X,Y)
    return classifier

def bagOfWordsExtractor(text):
    tweet = text["text"]
    return list(tweet.split())

def collectExamplesForPolitician(handle, featureList, Y, featureExtractor, hasDonor):
    try:
        with open("Tweets/" + handle + ".json") as json_file:
            data = json.load(json_file)
            for tweet in data:
                features = featureExtractor(tweet)
                allWords.update(dict.fromkeys(features))
                featureList.append(features)
                Y.append(hasDonor)
    except IOError:
        print ("Can't find file: Tweets/" + handle + ".json")

def collectExamplesForDonor(donor, politicianFile, featureExtractor, candidateToHandleMap):
    recipients = []
    featureList = []
    Y = []

    with open("donors/" + donor + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            recipients.append(row)

    with open(politicianFile) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            collectExamplesForPolitician(candidateToHandleMap[row[0]], featureList, Y, featureExtractor, 1 if row[0] in recipients else 0)

    return (featureList,Y)

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
    print("Collecting training examples...")
    trainExamples = collectExamplesForDonor(donor, trainFile, bagOfWordsExtractor, candidateToHandleMap)
    print("Formatting features...")
    X = getX(trainExamples[0])
    print("Learning predictor...")
    classifier = learnPredictor(X, trainExamples[1])
    print("Collecting test examples...")
    testExamples = collectExamplesForDonor(donor, testFile, bagOfWordsExtractor, candidateToHandleMap)
    print("Formatting features...")
    X = getX(testExamples[0])
    print("Scoring...")
    classifier.score(X, testExamples[1])

candidateToHandleMap = buildCandidateToHandleMap()
for donor in donors:
    runDonor(donor, candidateToHandleMap)
