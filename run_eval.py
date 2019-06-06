import algorithm as bias

params = {
        "NaiveBayes": False,
        "NaiveBayesCom": False,
        "NaiveBayesBer": False,
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

def runAll(modelParams, model, fullSpectrum, rL):
    print("Run all experiments for %s\n" % (model))
    modelParams["GuessBias"] = True
    runBiasExperiments(modelParams, model, "GuessBias", fullSpectrum, rL)
    modelParams["GuessBias"] = False
    modelParams["GuessPublisher"] = True
    runExperiments(modelParams, model, "GuessPublisher")
    modelParams["GuessPublisher"] = False
    modelParams["GuessBiasOfPublisher"] = True
    runExperiments(modelParams, model, "GuessBiasOfPublisher")

def runBiasExperiments(modelParams, model, predict, fullSpectrum, rL):
    if fullSpectrum:
        print("Run no ngrams, full spectrum for %s predicting %s\n" % (model, predict))
        bias.runExperiment(modelParams, fileName(model, False, False, predict))

    if rL:
        print("Run no ngrams, RL only for %s predicting %s\n" % (model, predict))
        modelParams["LeftRightOnly"] = True
        bias.runExperiment(modelParams, fileName(model, True, False, predict))

    if fullSpectrum:
        print("Run with ngrams, full spectrum for %s predicting %s\n" % (model, predict))
        modelParams["Ngrams"] = True
        modelParams["LeftRightOnly"] = False
        bias.runExperiment(modelParams, fileName(model, False, True, predict))

    if rL:
        print("Run with ngrams, RL only for %s predicting %s\n" % (model, predict))
        modelParams["LeftRightOnly"] = True
        bias.runExperiment(modelParams, fileName(model, True, True, predict))


def runOtherExperiments(modelParams, model, predict):
    print("Run no ngrams, full spectrum for %s predicting %s\n" % (model, predict))
    bias.runExperiment(modelParams, fileName(model, False, False, predict))

logisticRegressionParams = params.copy()
runAll(logisticRegressionParams, "LogisticRegression", True, True)

naiveBayesParams = params.copy()
naiveBayesParams["NaiveBayesCom"] = True
runAll(naiveBayesParams, "NaiveBayes", True, False)

naiveBayesLaplaceParams = params.copy()
naiveBayesLaplaceParams["NaiveBayesCom"] = True
naiveBayesLaplaceParams["LaplaceSmoothing"] = True
runAll(naiveBayesLaplaceParams, "LaplaceNaiveBayes", True, False)

naiveBayesParams = params.copy()
naiveBayesParams["NaiveBayesBer"] = True
runAll(naiveBayesParams, "NaiveBayes", False, True)

naiveBayesLaplaceParams = params.copy()
naiveBayesLaplaceParams["NaiveBayesBer"] = True
naiveBayesLaplaceParams["LaplaceSmoothing"] = True
runAll(naiveBayesLaplaceParams, "LaplaceNaiveBayes", False, True)
