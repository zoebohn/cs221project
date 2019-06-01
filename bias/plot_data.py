import matplotlib
import matplotlib.pyplot as plt
import re

SUBDIR = "out"

bgbox = {'color': "white", 'pad':0.5}

DARK_RED = '#7a1616'

def fileName(model, isLR, nGrams, predict):
    LrOrFull = "LeftRightOnly" if isLR else "FullSpectrum"
    nGramStr = "nGrams" if nGrams else "noNgrams"
    return "%s/%s_%s_%s_%s" % (SUBDIR, model, LrOrFull, nGramStr, predict)

models = ["LogisticRegression", "NaiveBayes", "LaplaceNaiveBayes"]
modelNames = ["Random", "SGD\n(Baseline)", "Naive Bayes", "Naive Bayes +\nLaplace Smoothing"]

def extractDataFromFiles(isLr, predict):
    trainArr = []
    testArr = []
    # random
    if isLr:
        trainArr.append(50.0)
        testArr.append(50.0)
    else:
        testArr.append(20.0)
        trainArr.append(20.0)
    for model in models:
        with open(fileName(model, isLr, False, predict), "r") as f:
            lines = f.readlines()
            print(lines)
            trainM = re.match("train: (.*)", lines[0])
            trainArr.append(float(trainM.group(1)) * 100.0)
            testM = re.match("test: (.*)", lines[1])
            testArr.append(float(testM.group(1)) * 100.0)
    return trainArr, testArr


def makeBiasFullSpectrumPlot():
    trainArr, testArr = extractDataFromFiles(False, "GuessBias")
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(models) + 1), testArr, color=DARK_RED)
    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_y() + bars[i].get_height() + bars[0].get_height() / 20, str(round(testArr[i])), ha='center', va='center', bbox=bgbox)
    plt.title("Accuracy of Article Title Classification (Full Spectrum)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 40)
    plt.xticks(range(len(modelNames)), modelNames)
    plt.savefig('figs/bias_fullspectrum.png')

def makeBiasLROnlyPlot():
    trainArr, testArr = extractDataFromFiles(True, "GuessBias")
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(models) + 1), testArr, color=DARK_RED)
    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_y() + bars[i].get_height() + bars[0].get_height() / 30, str(round(testArr[i])), ha='center', va='center', bbox=bgbox)
    plt.title("Accuracy of Article Title Classification (Left or Right Only)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 70)
    plt.xticks(range(len(modelNames)), modelNames)
    plt.savefig('figs/bias_leftright.png')

makeBiasFullSpectrumPlot()
makeBiasLROnlyPlot()
