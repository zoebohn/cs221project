import csv
import json
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

ps = PorterStemmer()
NEUTRAL_PARTISAN_ONLY = False
LEFT_RIGHT_ONLY = True
NAIVE_BAYES = True
NGRAMS = True
assert(not (NEUTRAL_PARTISAN_ONLY and LEFT_RIGHT_ONLY))

def stemmedBagOfWordsExtractor(text):
    return [ps.stem(word) for word in text.split()]

def bagOfWordsExtractor(text):
    return list(text.split())

def nGramExtractor(text):
    ngram_length = 2
    words = list(text.split())
    ngrams = []
    for i in range(len(words) - ngram_length + 1):
        ngram = words[i]
        ngrams.append(ngram)
        for j in range(i + 1, i + ngram_length):
            ngram += " "
            ngram += words[j]
        ngrams.append(ngram)
    return ngrams

def loadData():

    X_raw = [] # title
    #Y = [] # publisher_score
    publisher_to_title_map = {}

    publisher_to_score_map = {}
    matched_publishers = set()

    with open("news_bias.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            publisher = row[0]
            score = None
            if NEUTRAL_PARTISAN_ONLY:
                score = 0 if int(row[1]) == 0 else 1
            if LEFT_RIGHT_ONLY:
                if int(row[1]) == 0:
                    continue
                score = -1 if int(row[1]) > 0 else 1
            else:
                score = int(row[1])
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
                if publisher not in publisher_to_title_map:
                    publisher_to_title_map[publisher] = []
                publisher_to_title_map[publisher].append(title)
                #X_raw.append(title)
                #Y.append(publisher_score)
                matched_publishers.add(publisher)
    
    print("Matched " + str(len(matched_publishers)) + " publishers out of possible " + str(len(publisher_to_score_map.keys())))

    buckets = [0,0,0,0,0]
    pub_file = open("matched_pubs.txt", "w")
    for publisher in matched_publishers:
        buckets[publisher_to_score_map[publisher] + 2] += 1
        pub_file.write(publisher + str(publisher_to_score_map[publisher]) + "\n")
    pub_file.close()
    print(buckets)


    return publisher_to_title_map, publisher_to_score_map

def buildFeatureVectors(X_raw):
    X_map = []
    featureExtractor = None
    if NGRAMS:
        featureExtractor = nGramExtractor
    else:
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

        classifier = None
        if NAIVE_BAYES:
            print("Using Naive Bayes")
            classifier = MultinomialNB()
            classifier.fit(X_train, Y_train)
        else:
            classifier = linear_model.SGDClassifier()
            classifier.fit(X_train, Y_train)
        
        print("Classified data")

        print(classifier.score(X_train, Y_train))
        print("Scored train data")
        print(classifier.score(X_test, Y_test))
        print("Scored testing data")
        predictions = classifier.predict(X_test)
        buckets_correct = [0, 0, 0, 0, 0]
        buckets_guessed = [0, 0, 0, 0, 0]
        buckets_total = [0, 0, 0, 0, 0]
        for i, prediction in enumerate(predictions):
            score = 1 if prediction == Y_test[i] else 0
            buckets_correct[prediction + 2] += score
            buckets_guessed[prediction + 2] += 1
            buckets_total[Y_test[i] + 2] += 1

        print(buckets_correct)
        print(buckets_guessed)
        print(buckets_total)

def main():
    publisher_to_title, publisher_to_score = loadData()
    print("Loaded data")
    train_X_raw = []
    train_Y = []
    test_X_raw = []
    test_Y = []

    # TODO - randomly shuffle order
    publisher_list = list(publisher_to_title.keys())
    train_test_boundary = int(len(publisher_list)/2.0)
    train_X = []
    for i in range(train_test_boundary):
        train_X_raw += publisher_to_title[publisher_list[i]]
        train_Y += [publisher_to_score[publisher_list[i]]] * len(publisher_to_title[publisher_list[i]])
    for i in range(train_test_boundary):
        test_X_raw += publisher_to_title[publisher_list[i + train_test_boundary]]
        test_Y += [publisher_to_score[publisher_list[i + train_test_boundary]]] * len(publisher_to_title[publisher_list[i + train_test_boundary]])

    buckets = [0,0,0,0,0]
    for i in range(train_test_boundary):
        publisher = publisher_list[i]
        buckets[publisher_to_score[publisher] + 2] += 1
    print ("Train buckets:")
    print(buckets)
    
    print("Split data into train and test")
    splitPoint = len(train_X_raw)
    combined_X = buildFeatureVectors(train_X_raw + test_X_raw)
    train_X = combined_X[:splitPoint]
    test_X = combined_X[splitPoint:]
    print("Built feature vectors")
    print("len train_X: " + str(len(train_X)) + ", len train_Y: " + str(len(train_Y)))
    print("len test_X: " + str(len(test_X)) + ", len test_Y: " + str(len(test_Y)))

    evaluate(train_X, train_Y, test_X, test_Y)

main()
