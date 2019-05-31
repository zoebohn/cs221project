import news_bias_baseline as bias

params = {
        "NaiveBayes": False,
        "NaiveBayesCom": False,
        "LaplaceSmoothing": False,
        "NeutralPartisanOnly": False,
        "LeftRightOnly": False,
        "Ngrams": False,
        "GuessBias": False,
        "GuessPublisher": False,
        "GuessBiasOfPublisher": False
        }

SUBDIR = "out"

def fileName(model, isLR, nGrams, predict):
    LrOrFull = "LeftRightOnly" if isLR else "FullSpectrum"
    nGramStr = "nGrams" if nGrams else "noNgrams"
    return "%s/%s_%s_%s_%s" % (SUBDIR, model, LrOrFull, nGramStr, predict)

def runAll(modelParams, model):
    print("Run all experiments for %s\n" % (model))
    modelParams["GuessBias"] = True
    runExperiments(modelParams, True, model, "GuessBias")
    modelParams["GuessBias"] = False
    modelParams["GuessPublisher"] = True
    #runExperiments(modelParams, False, model, "GuessPublisher")
    modelParams["GuessPublisher"] = False
    modelParams["GuessBiasOfPublisher"] = True
    #runExperiments(modelParams, True, model, "GuessBiasOfPublisher")

def runExperiments(modelParams, includeRLOnly, model, predict):
#    print("Run no ngrams, full spectrum for %s predicting %s\n" % (model, predict))
#    bias.runExperiment(modelParams, fileName(model, False, False, predict))

#    if includeRLOnly:
#        print("Run no ngrams, RL only for %s predicting %s\n" % (model, predict))
#        modelParams["LeftRightOnly"] = True
#        bias.runExperiment(modelParams, fileName(model, True, False, predict))

    print("Run with ngrams, full spectrum for %s predicting %s\n" % (model, predict))
    modelParams["Ngrams"] = True
    modelParams["LeftRightOnly"] = False
    bias.runExperiment(modelParams, fileName(model, False, True, predict))

    if includeRLOnly:
        print("Run with ngrams, RL only for %s predicting %s\n" % (model, predict))
        modelParams["LeftRightOnly"] = True
        bias.runExperiment(modelParams, fileName(model, True, True, predict))

logisticRegressionParams = params.copy()
runAll(logisticRegressionParams, "LogisticRegression")

naiveBayesParams = params.copy()
naiveBayesParams["NaiveBayes"] = True
runAll(naiveBayesParams, "NaiveBayes")

naiveBayesLaplaceParams = params.copy()
naiveBayesLaplaceParams["NaiveBayes"] = True
naiveBayesLaplaceParams["LaplaceSmoothing"] = True
runAll(naiveBayesLaplaceParams, "LaplaceNaiveBayes")
