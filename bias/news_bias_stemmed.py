import csv
import json
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

def learnPredictor(X, Y):
    classifier = linear_model.SGDClassifier()
    classifier.fit(X,Y)
    return classifier

def bagOfWordsExtractor(text):
    return [ps.stem(word) for word in text.split()]
    #return list(text.split())


def loadData():

    X_raw = [] # title
    Y = [] # publisher_score

    publisher_to_score_map = {}
    matched_publishers = set()

    with open("news_bias.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            publisher = row[0]
            score = 0 if int(row[1]) == 0 else 1
            publisher_to_score_map[publisher] = score

    with open("news-articles.csv", "rU") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            title = row[1]
            publisherCluttered = row[3]
            publisher = None
            if publisherCluttered.find('(') > 1:
                publisher = publisherCluttered[:publisherCluttered.find('(') - 2]
            else:
                publisher = publisherCluttered
            publisher_score = publisher_to_score_map.get(publisher, None)
            if publisher_score is not None:
                X_raw.append(title)
                Y.append(publisher_score)
                matched_publishers.add(publisher)
    
    print("Matched " + str(len(matched_publishers)) + " publishers out of possible " + str(len(publisher_to_score_map.keys())))

    buckets = [0,0,0,0,0]
    pub_file = open("matched_pubs.txt", "w")
    for publisher in matched_publishers:
        buckets[publisher_to_score_map[publisher] + 2] += 1
        pub_file.write(publisher + str(publisher_to_score_map[publisher]))
    pub_file.close()
    print(buckets)


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
        print(str(len(X_train)) + " data points in our training set")
        classifier = learnPredictor(X_train, Y_train)
        print("Classified data")

        print(classifier.score(X_test, Y_test))
        print("Scored data")
        predictions = classifier.predict(X_test)
        buckets_correct = [0, 0, 0, 0, 0]
        buckets_guessed = [0, 0, 0, 0, 0]
        buckets_total = [0, 0, 0, 0, 0]
        for i, prediction in enumerate(predictions):
            score = 1 if prediction == Y_test[i] else 0
            buckets_correct[prediction + 2] += score
            buckets_guessed[prediction + 2] += 1
            buckets_total[Y_test[i] + 2] += 1

        print("buckets correct: " + str(buckets_correct))
        print("buckets guessed: " + str(buckets_guessed))
        print("buckets total: " + str(buckets_total))

def main():
    X_raw, Y = loadData()
    print("Loaded data")
    X = buildFeatureVectors(X_raw)
    print("Built feature vectors")

    # TODO - randomly shuffle order 
    train_test_boundary = int(len(Y)/2.0)
    train_X = X[:train_test_boundary]
    test_X = X[train_test_boundary:]
    train_Y = Y[:train_test_boundary]
    test_Y = Y[train_test_boundary:]
    print("Divided data into train and test")

    evaluate(train_X, train_Y, test_X, test_Y)

main()
