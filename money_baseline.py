import csv
import json
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer

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
    # donor name -> donation amount
    donationMap = {}
    with open("congress_tweet_handles_sorted.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            name = row[0]
            handle = row[1]
            candidateNameList.append(name)
            candidateTweetHandleList.append(handle)

    with open("money_raised/money_raised.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            donationMap[row[0]] = row[1]

    return candidateNameList, candidateTweetHandleList, donationMap

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
        
def evaluate(candidateFeatureVectorList, donationMap, train, test):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        for candidateId, candidate in enumerate(train):
            if candidate in donationMap:
                Y_train.append(donationMap[candidate])
            else:
                print("(train) Couldn't find: " + candidate)
                continue
            X_train.append(candidateFeatureVectorList[candidateId])
        print("Gathered train data")

        for candidateId, candidate in enumerate(test):
            if candidate in donationMap:
                Y_test.append(donationMap[candidate])
            else:
                print("(test) Couldn't find: " + candidate)
                continue 
            X_test.append(candidateFeatureVectorList[candidateId])
        print("Gathered test data")

        classifier = learnPredictor(X_train, Y_train)
        print("Classified data")

        print(classifier.score(X_test, Y_test))
        print("Scored data")
        predictions = classifier.predict(X_test)
        avg_error_rate = 0.0
        for i, prediction in enumerate(predictions):
            avg_error_rate += abs((Y_test[i] - prediction) / Y_test[i])
        avg_error_rate /= float(len(predictions))
        print("Average error rate: " + str(avg_error_rate))

def main():
    candidateNameList, candidateTweetHandleList, donationMap = loadData()
    print("Loaded data")
    candidateFeatureVectorList = buildFeatureVectors(candidateTweetHandleList)
    print("Loaded tweets")

    numCandidates = len(candidateNameList)
    # TODO - randomly shuffle order of candidates in file
    train_test_boundary = int(numCandidates/2.0)
    train = candidateNameList[:train_test_boundary]
    test = candidateNameList[train_test_boundary:]
    print("Divided data into train and test")

    evaluate(candidateFeatureVectorList, donationMap, train, test)

main()
