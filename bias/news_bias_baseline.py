import csv
import json
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer

def learnPredictor(X, Y):
    classifier = linear_model.SGDClassifier()
    classifier.fit(X,Y)
    return classifier

def bagOfWordsExtractor(text):
    return list(text.split())


def loadData():

    X_raw = [] # title
    Y = [] # publisher_score

    publisher_to_score_map = {}
    matched_publishers = set()

    with open("TODO") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            publisher = row[0]
            score = row[1]
            publisher_to_score_map[publisher] = score

    with open("news-articles.csv", "rU") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            title = row[1]
            publisher = row[3]
            score = publisher_to_score_map.get(publisher, None)
            if score is not None:
                X_raw.append(title)
                Y.append(publisher_score)
                matched_publishers.add(publisher)
    
    print "Matched " + len(matched_publishers) + " publishers"

    return X_raw, Y 

def buildFeatureVectors(X_raw):
    X_map = []
    featureExtractor = bagOfWordsExtractor

    for x in X_raw:
        featureMap = {}
        features = featureExtractor(x)
        for feature in features:
            featureMap[feature] = 1
        X_map.append(featureMap)
        
    dv = DictVectorizer(sparse=False)
    vectors = dv.fit_transform(X_map)
    return vectors
        
def evaluate(X_train, Y_train, X_test, Y_test):
        classifier = learnPredictor(X_train, Y_train)
        print "Classified data"

        print classifier.score(X_test, Y_test)
        print "Scored data"
        predictions = classifier.predict(X_test)
        one_yay = 0
        zero_yay = 0
        zero_miss = 0
        one_miss = 0
        for i, prediction in enumerate(predictions):
            if Y_test[i] == 1:
                if prediction == Y_test[i]:
                    one_yay += 1
                else:
                    one_miss += 1
            else:
                if prediction == Y_test[i]:
                    zero_yay += 1
                else:
                    zero_miss += 1
        print "Correctly classified " + str(one_yay) + " as getting donations out of a total of " + str(one_yay + one_miss) + " who did get donations in this test set."
        print "Correctly classified " + str(zero_yay) + " as NOT getting donations out of a total of " + str(zero_yay + zero_miss) + " who did NOT get donations in this test set."

def main():
    X_raw, Y = loadData()
    print "Loaded data"
    X = buildFeatureVectors(X_raw)
    print "Built feature vectors"

    # TODO - randomly shuffle order 
    train_test_boundary = int(len(Y)/2.0)
    train_X = X[:train_test_boundary]
    test_X = X[train_test_boundary:]
    train_Y = Y[:train_test_boundary]
    test_Y = Y[train_test_boundary:]
    print "Divided data into train and test"

    evaluate(train_X, train_Y, test_X, test_Y)

main()
