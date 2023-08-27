import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import time

uniqueTags = ['NNP',',','CD','NNS','JJ','MD','VB','DT','NN','IN','.','VBZ','VBG','CC','VBD','VBN','RB','TO','PRP','RBR','WDT','VBP','RP','PRP$','JJS','POS','``','EX',"''",'WP',':','JJR','WRB','$','NNPS','WP$','-LRB-','-RRB-','PDT','RBS','FW','UH','SYM','LS','#']

initialPOSProbabilities = {'NNP': 0.09605139815479748, ',': 0.05095960398861961, 'CD': 0.03823724502381879, 'NNS': 0.06343527812344109, 'JJ': 0.06462484719245254, 'MD': 0.010346509957844304, 'VB': 0.027945553917081006, 'DT': 0.08636709991831991, 'NN': 0.13982534714037465, 'IN': 0.10389049386302962, '.': 0.041534050729364815, 'VBZ': 0.023004182678339428, 'VBG': 0.015730817513526552, 'CC': 0.025016034513948657, 'VBD': 0.03103733711948865, 'VBN': 0.021192967837780057, 'RB': 0.0324757837725237, 'TO': 0.023529347271939876, 'PRP': 0.018381857153037785, 'RBR': 0.00183643151206837, 'WDT': 0.004598205230814773, 'VBP': 0.013513943174778944, 'RP': 0.002757388210657881, 'PRP$': 0.008758956029799527, 'JJS': 0.002046935900317401, 'POS': 0.009082387251327987, '``': 0.007435628964088171, 'EX': 0.0009132820594345984, "''": 0.007260208640547311, 'WP': 0.002505221495567896, ':': 0.005131044463570132, 'JJR': 0.003479900668241795, 'WRB': 0.002247572895367259, '$': 0.007605567402518378, 'NNPS': 0.0027464244404365773, 'WP$': 0.00018199858567364145, '-LRB-': 0.001430772013880133, '-RRB-': 0.001448314046234219, 'PDT': 0.0003650935483694133, 'RBS': 0.00047692400462671106, 'FW': 0.0002455884529572029, 'UH': 9.538480092534221e-05, 'SYM': 6.030073621717036e-05, 'LS': 5.15297200401274e-05, '#': 0.00013923988181055701}

file = open('hmm.json')
data = json.load(file)
file.close()
emissionParam, transitionParam = data["emission"], data["transition"]

def validVocab():
    with open("vocab.txt", "r") as file:
        vocabWords = file.readlines() 
    validVocabWords = []

    for data in vocabWords:
        word = data.split("\t")
        validVocabWords.append(word[0])
    validVocabWords = validVocabWords[1:]
    t2 = time.time()
    return validVocabWords

def greedyDecoding(sentence):
    t1 = time.time()
    calculatedTags = []
    
    calculatedProbilities = [0]*len(uniqueTags)
    for i in range(len(calculatedProbilities)):
        emmissionKey = uniqueTags[i] + " <-> " + sentence[0]
        if emmissionKey not in emissionParam:
            calculatedProbilities[i] = np.float64(0)
        else:
            calculatedProbilities[i] = np.float64(emissionParam[emmissionKey]*initialPOSProbabilities[uniqueTags[i]])
    
    maxProbabilityIdx = calculatedProbilities.index(max(calculatedProbilities))
    calculatedTags.append(uniqueTags[maxProbabilityIdx])
    
    for i in range(1, len(sentence)):
        word = sentence[i]
        calculatedProbilities = [0]*len(uniqueTags)
        for j in range(len(calculatedProbilities)):
            emmisionKey = uniqueTags[j] + " <-> " + word
            transmissionKey = calculatedTags[-1] + " <-> " + uniqueTags[j]
            if emmisionKey not in emissionParam or transmissionKey not in transitionParam:
                calculatedProbilities[j] = np.float64(0)
            else:
                calculatedProbilities[j] = np.float64(emissionParam[emmisionKey]*transitionParam[transmissionKey])
        maxProbabilityIdx = calculatedProbilities.index(max(calculatedProbilities))
        calculatedTags.append(uniqueTags[maxProbabilityIdx])
    
    return calculatedTags


def findTestPOSTags():
    with open("data/test", "r") as file:
        testDataSet = file.readlines()
    testDataSet += ["\n"]

    validVocabWords = validVocab()
    sentencesList, actualSentenceList, tempSentencesList, tempActualSentence = [], [], [], []

    for data in testDataSet:
        words = data.split("\t")
        if len(words) > 1:
            w = words[1][:-1]
            tempActualSentence.append(words[1][:-1])
            if w not in validVocabWords:
                w = "<unk>"
            tempSentencesList.append(w)
        else:
            sentencesList.append(tempSentencesList)
            actualSentenceList.append(tempActualSentence)
            tempSentencesList, tempActualSentence = [], []

    predictedTestTags = []
    for i in range(len(sentencesList)):
        calculatedTags = greedyDecoding(sentencesList[i])
        predictedTestTags.append(calculatedTags)
    
    finalTestSentences = []
    for sentence, tags in tqdm(zip(actualSentenceList, predictedTestTags)):
        k = 1
        for i in range(len(sentence)):
            testSentence = str(k) + "\t" + sentence[i] + "\t" + tags[i] + "\n"
            k += 1
            finalTestSentences.append(testSentence)
        finalTestSentences.append("\n")

    greedyTxtFile = open("greedy.out",  "w+")
    for data in finalTestSentences:
        greedyTxtFile.write(data)
    greedyTxtFile.close()

findTestPOSTags()