import csv
import json
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer

donorList = ["bloomberg", "fahr", "koch", "las_vegas_sands", "nra", "paloma", "planned_parenthood", "uline"]

def learnPredictor(X, Y):
    classifier = linear_model.SGDClassifier()
    classifier.fit(X,Y)
    return classifier

def bagOfWordsExtractor(text):
    tweet = text["text"]
    return list(tweet.split())


def loadData():

    # index -> "last name, first name"
    candidateNameList = []
    # index -> "handle"
    candidateTweetHandleList = []
    # donor name -> set of candidate ID
    donorMap = {}
    with open("congress_tweet_handles_sorted.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            name = row[0]
            handle = row[1]
            candidateNameList.append(name)
            candidateTweetHandleList.append(handle)

    for donor in donorList:
        candidateIdSet = set()
        with open("donors/" + donor + ".csv") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                nameCluttered = row[0]
                name = nameCluttered[:nameCluttered.index('(') - 1]
                if name not in candidateNameList:
                    continue
                else:
                    candidateIdSet.add(candidateNameList.index(name))
        donorMap[donor] = candidateIdSet

    return candidateNameList, candidateTweetHandleList, donorMap

def buildFeatureVectors(candidateTweetHandleList):
    candidateFeaturesList = []
    featureExtractor = bagOfWordsExtractor

    for candidateTweetHandle in candidateTweetHandleList:
        candidateFeatures = {} 
        try:
            with open("Tweets_Final/" + candidateTweetHandle + ".json") as json_file:
                data = json.load(json_file)
                for tweet in data:
                    features = featureExtractor(tweet)
                    for feature in features:
                        candidateFeatures[feature] = 1
        except IOError:
            print ("Can't find file: Tweets_Final/" + candidateTweetHandle + ".json")
        candidateFeaturesList.append(candidateFeatures) 
    dv = DictVectorizer(sparse=False)
    vectors = dv.fit_transform(candidateFeaturesList)
    return vectors
        
def evaluate(candidateFeatureVectorList, donorMap, train, test):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        for candidateId, candidate in enumerate(train):
            X_train.append(candidateFeatureVectorList[candidateId])
            Y_train.append(1 if candidateId in donorMap[donor] else 0)
        print "Gathered train data"

        for candidateId, candidate in enumerate(test):
            X_test.append(candidateFeatureVectorList[candidateId])
            Y_test.append(1 if candidateId in donorMap[donor] else 0)
        print "Gathered test data"

        classifier = learnPredictor(X_train, Y_train)
        print "Classified data"

        classifier.score(X_test, Y_test)
        print "Scored data"

def main():
    candidateNameList, candidateTweetHandleList, donorMap = loadData()
    print "Loaded data"
    candidateFeatureVectorList = buildFeatureVectors(candidateTweetHandleList)
    print "Loaded tweets"

    numCandidates = len(candidateNameList)
    # TODO - randomly shuffle order of candidates in file
    train_test_boundary = int(numCandidates/2.0)
    train = candidateNameList[:train_test_boundary]
    test = candidateNameList[train_test_boundary:]
    print "Divided data into train and test"

    for donor in donorList:
        evaluate(candidateFeatureVectorList, donorMap, train, test)

main()
