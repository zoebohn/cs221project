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
    # donor name -> map of candidate ID to donation amounts
    donorMap = {}
    with open("congress_tweet_handles_sorted.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            name = row[0]
            handle = row[1]
            candidateNameList.append(name)
            candidateTweetHandleList.append(handle)

    for donor in donorList:
        donationAmounts = {}
        with open("donors/" + donor + ".csv") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                nameCluttered = row[0]
                name = nameCluttered[:nameCluttered.index('(') - 1]
                if name not in candidateNameList:
                    continue
                else:
                    donationAmounts[candidateNameList.index(name)] = float(row[2][1:].replace(',',''))
        donorMap[donor] = donationAmounts 

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
        
def evaluate(donor, candidateFeatureVectorList, donorMap, train, test):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        for candidateId, candidate in enumerate(train):
            X_train.append(candidateFeatureVectorList[candidateId])
            if candidateId in donorMap[donor]:
                Y_train.append(donorMap[donor][candidateId])
            else:
                Y_train.append(0.0)
        print("Gathered train data")

        for candidateId, candidate in enumerate(test):
            X_test.append(candidateFeatureVectorList[candidateId])
            if candidateId in donorMap[donor]:
                Y_test.append(donorMap[donor][candidateId])
            else:
                Y_test.append(0.0)
        print("Gathered test data")

        classifier = learnPredictor(X_train, Y_train)
        print("Classified data")

        print(classifier.score(X_test, Y_test))
        print("Scored data")
        predictions = classifier.predict(X_test)
        avg_error_rate = 0.0 
        avg_total_error = 0.0 
        for i, prediction in enumerate(predictions):
            avg_total_error += (abs(Y_test[i] - prediction) / float(len(predictions)))
            if Y_test[i] == 0:
                avg_error_rate += abs(Y_test[i] - prediction)
            else:   
                avg_error_rate += abs(Y_test[i] - prediction) / Y_test[i]
            print("wanted " + str(Y_test[i]) + " and got " + str(prediction))
        avg_error_rate /= float(len(predictions))
        print("Average error rate: " + str(avg_error_rate))
        print("Total error: " + str(avg_total_error))


def main():
    candidateNameList, candidateTweetHandleList, donorMap = loadData()
    print("Loaded data")
    candidateFeatureVectorList = buildFeatureVectors(candidateTweetHandleList)
    print("Loaded tweets")

    numCandidates = len(candidateNameList)
    # TODO - randomly shuffle order of candidates in file
    train_test_boundary = int(numCandidates/2.0)
    train = candidateNameList[:train_test_boundary]
    test = candidateNameList[train_test_boundary:]
    print("Divided data into train and test")

    for donor in donorList:
        evaluate(donor, candidateFeatureVectorList, donorMap, train, test)

main()
