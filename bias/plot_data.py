import matplotlib
import matplotlib.pyplot as plt
import re
import brewer2mpl

bmap1 = brewer2mpl.get_map('Set1', 'Qualitative', 7)
bmap2 = brewer2mpl.get_map('Dark2', 'Qualitative', 7)
hash_colors = bmap1.mpl_colors
mix_colors = bmap2.mpl_colors

SUBDIR = "out"

bgbox = {'color': "white", 'pad':0.5}

DARK_RED = '#7a1616'

def fileName(model, isLR, nGrams, predict):
    LrOrFull = "LeftRightOnly" if isLR else "FullSpectrum"
    nGramStr = "nGrams" if nGrams else "noNgrams"
    return "%s/%s_%s_%s_%s" % (SUBDIR, model, LrOrFull, nGramStr, predict)

models = ["LogisticRegression", "NaiveBayes", "LaplaceNaiveBayes"]
modelNames = ["Random", "SGD\n(Baseline)", "Naive Bayes", "Naive Bayes +\nLaplace Smoothing"]
modelNamesOracle = ["Random", "SGD\n(Baseline)", "Naive Bayes", "Naive Bayes +\nLaplace Smoothing", "Oracle"]
modelNamesBigrams = ["Random", "SGD\n(Baseline)", "SGD +\nBigrams", "Naive Bayes", "Naive Bayes+\nBigrams", "Naive Bayes +\nLaplace Smoothing", "Naive Bayes +\nLaplace Smoothing +\nBigrams"]
modelNamesPublisher = ["Random", "SGD", "Naive Bayes", "Naive Bayes +\nLaplace Smoothing"]

def extractDataFromFiles(isLr, nGrams, predict):
    trainArr = []
    testArr = []
    # random
    if predict is "GuessPublisher":
        trainArr.append(1.0)
        testArr.append(1.0)
    elif isLr:
        trainArr.append(50.0)
        testArr.append(50.0)
    else:
        testArr.append(20.0)
        trainArr.append(20.0)
    for model in models:
        with open(fileName(model, isLr, nGrams, predict), "r") as f:
            lines = f.readlines()
            print(lines)
            trainM = re.match("train: (.*)", lines[0])
            trainArr.append(float(trainM.group(1)) * 100.0)
            testM = re.match("test: (.*)", lines[1])
            testArr.append(float(testM.group(1)) * 100.0)
    return trainArr, testArr


def makeBiasFullSpectrumPlot():
    trainArr, testArr = extractDataFromFiles(False, False, "GuessBias")
    fig, ax = plt.subplots()
    bars = plt.bar(range(len(models) + 1), testArr, color=mix_colors[0])
    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_y() + bars[i].get_height() - bars[0].get_height() / 15, str(round(testArr[i])), ha='center', va='center', color='white')
    plt.title("Accuracy of Article Title Classification (Full Spectrum)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 40)
    plt.xticks(range(len(modelNames)), modelNames)
    plt.savefig('figs/bias_fullspectrum.png')

def makeBiasOracleFullSpectrumPlot():
    trainArr, testArr = extractDataFromFiles(False, False, "GuessBias")
    testArr.append(trainArr[1])
    fig, ax = plt.subplots()
    bars = plt.bar(range(len(models) + 2), testArr, color=mix_colors[0])
    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_y() + bars[i].get_height() - bars[0].get_height() / 4, str(round(testArr[i])), ha='center', va='center', color='white')
    plt.title("Accuracy of Article Title Classification with Oracle (Full Spectrum)")
    plt.ylabel("Accuracy (%)")
    #plt.ylim(0, 40)
    plt.xticks(range(len(modelNamesOracle)), modelNamesOracle, rotation='vertical')
    plt.subplots_adjust(bottom=0.35)
    plt.savefig('figs/bias_oracle_fullspectrum.png')

def makeBiasLROnlyPlot():
    trainArr, testArr = extractDataFromFiles(True, False, "GuessBias")
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(models) + 1), testArr, color=mix_colors[1])
    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_y() + bars[i].get_height() - bars[0].get_height() / 20, str(round(testArr[i])), ha='center', va='center', color='white')
    plt.title("Accuracy of Article Title Classification (Left or Right Only)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 75)
    plt.xticks(range(len(modelNames)), modelNames)
    plt.savefig('figs/bias_leftright.png')

def makeBiasOracleLROnlyPlot():
    trainArr, testArr = extractDataFromFiles(True, False, "GuessBias")
    fig, ax = plt.subplots()
    testArr.append(trainArr[1])
    bars = plt.bar(range(len(models) + 2), testArr, color=mix_colors[1])
    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_y() + bars[i].get_height() - bars[0].get_height() / 8, str(round(testArr[i])), ha='center', va='center', color='white')
    plt.title("Accuracy of Article Title Classification with Oracle (Left or Right Only)")
    plt.ylabel("Accuracy (%)")
    #plt.ylim(0, 40)
    plt.xticks(range(len(modelNamesOracle)), modelNamesOracle, rotation='vertical')
    plt.subplots_adjust(bottom=0.35)
    plt.savefig('figs/bias_oracle_leftright.png')



def makeBigramsBiasFullSpectrumPlot():
    trainArr, testArr = extractDataFromFiles(False, True, "GuessBias")
    trainArr2, testArr2 = extractDataFromFiles(False, False, "GuessBias")
    finalArr = [testArr[0], testArr2[1], testArr[1], testArr2[2], testArr[2], testArr2[3], testArr[3]]
    fig, ax = plt.subplots()
    bars = plt.bar(range(len(finalArr)), finalArr, color=mix_colors[2])
    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_y() + bars[i].get_height() - bars[0].get_height() / 8, str(round(finalArr[i])), ha='center', va='center', color='white')
    plt.title("Accuracy of Article Title Classification with Bigrams (Full Spectrum)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 40)
    plt.xticks(range(len(modelNamesBigrams)), modelNamesBigrams, rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    plt.savefig('figs/bias_ngrams_fullspectrum.png')

def makeBigramsBiasLROnlyPlot():
    trainArr, testArr = extractDataFromFiles(True, True, "GuessBias")
    trainArr2, testArr2 = extractDataFromFiles(True, False, "GuessBias")
    finalArr = [testArr[0], testArr2[1], testArr[1], testArr2[2], testArr[2], testArr2[3], testArr[3]]
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(finalArr)), finalArr, color=mix_colors[3])
    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_y() + bars[i].get_height() - bars[0].get_height() / 8, str(round(finalArr[i])), ha='center', va='center', color='white')
    plt.title("Accuracy of Article Title Classification with Bigrams (Left or Right)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 75)
    plt.xticks(range(len(modelNamesBigrams)), modelNamesBigrams, rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    plt.savefig('figs/bias_ngrams_leftright.png')

def makePublisherPlot():
    trainArr, testArr = extractDataFromFiles(False, False, "GuessPublisher")
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(models) + 1), testArr, color=mix_colors[4])
    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_y() + bars[i].get_height() + bars[1].get_height() / 25, str(round(testArr[i])), ha='center', va='center')
    plt.title("Accuracy of Predicting Publisher from Article Title")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 45)
    plt.xticks(range(len(modelNamesPublisher)), modelNamesPublisher)
    plt.savefig('figs/publisher.png')

def makeBiasOfPublisherFullSpectrumPlot():
    trainArr, testArr = extractDataFromFiles(False, False, "GuessBiasOfPublisher")
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(models) + 1), testArr, color=mix_colors[5])
    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_y() + bars[i].get_height() - bars[0].get_height() / 8, str(round(testArr[i])), ha='center', va='center', color='white')
    plt.title("Accuracy of Predicting Publisher Bias (Full Spectrum)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 45)
    plt.xticks(range(len(modelNamesPublisher)), modelNamesPublisher)
    plt.savefig('figs/biasPublisher_fullspectrum.png')

def makeBiasOfPublisherLROnlyPlot():
    trainArr, testArr = extractDataFromFiles(True, False, "GuessBiasOfPublisher")
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(models) + 1), testArr, color=hash_colors[1])
    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_y() + bars[i].get_height() - bars[0].get_height() / 16, str(round(testArr[i])), ha='center', va='center', color='white')
    plt.title("Accuracy of Predicting Publisher Bias (Left or Right Only)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 75)
    plt.xticks(range(len(modelNames)), modelNames)
    plt.savefig('figs/biasPublisher_leftright.png')




font = {'size': 11}
matplotlib.rc('font', **font)
makeBiasFullSpectrumPlot()
makeBiasOracleFullSpectrumPlot()
makeBiasLROnlyPlot()
makeBiasOracleLROnlyPlot()
makeBigramsBiasFullSpectrumPlot()
makeBigramsBiasLROnlyPlot()
makePublisherPlot()
makeBiasOfPublisherFullSpectrumPlot()
makeBiasOfPublisherLROnlyPlot()
