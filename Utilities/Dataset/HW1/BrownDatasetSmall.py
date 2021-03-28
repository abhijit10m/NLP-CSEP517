import os
import logging
from collections import Counter
from joblib import Parallel, delayed
import sys
logger = logging.getLogger('__main__')

devDataPath = os.path.join("corpus", "HW1_v2", "CSEP517-HW1-Data-Small", "brown.dev.txt")
trainDataPath = os.path.join("corpus", "HW1_v2", "CSEP517-HW1-Data-Small", "brown.train.txt")
testDataPath = os.path.join("corpus", "HW1_v2", "CSEP517-HW1-Data-Small", "brown.test.txt")

wordCount = Counter()
unkWords = set()
unkCandidates = set()
unk_symbol = "<unk>"
unk_threshold=1
max_unkwords=200
# unk_threshold=5
# max_unkwords=sys.maxsize

def unkTrain(line, max_unks):
    words = line.split(" ")
    for i, word in enumerate(words):
        if word in unkCandidates and wordCount[unk_symbol] < max_unks:
            words[i] = unk_symbol
            wordCount[unk_symbol]+=1
            unkWords.add(word)
    return words

def unk(line):
    words = line.split(" ")
    for i, word in enumerate(words):
        if word not in wordCount.keys() or word in unkWords:
            words[i] = unk_symbol
    return words

def find_unks(corpus):
    for i in range(len(corpus)):
        tokens = corpus[i].split(" ")
        for word in tokens:
            wordCount[word] += 1

    for word in wordCount.keys():
        if wordCount[word] <= unk_threshold:
            unkCandidates.add(word)

    logger.info("unk unkCandidates count : %d", len(unkCandidates))
    logger.info("unk unkCandidates : %s", unkCandidates)


def __loadRawData__(path):
    logger.info("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    return f.readlines()

def loadRawTrainData(unk_percentage = 0.001):
# def loadRawTrainData(unk_percentage = 1.0):
    trainCorpus = __loadRawData__(trainDataPath)
    find_unks(trainCorpus)
    word_count = sum(wordCount.values())
    max_unks = unk_percentage * word_count
    logger.info("Max unk words in train: %d", max_unks)
    return [unkTrain(line, max_unks) for line in trainCorpus]

def loadRawDevData():
    devCorpus = __loadRawData__(devDataPath)
    return [unk(line) for line in devCorpus]

def loadRawTestData():
    testCorpus = __loadRawData__(testDataPath)
    return [unk(line) for line in testCorpus]
