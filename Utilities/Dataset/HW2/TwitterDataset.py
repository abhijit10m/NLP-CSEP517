import os
import logging
from collections import Counter
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import json
from datetime import datetime

format = "%(asctime)s: %(message)s"
logfileName = os.path.join("logs", "HW2", datetime.now().strftime('log_%H_%M_%d_%m_%Y.log') )
logging.basicConfig(filename=logfileName, filemode='a', format=format, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger('__HW2__')

devDataPath = os.path.join("corpus", "HW2", "CSEP517-HW2-Data", "twt.dev.json")
trainDataPath = os.path.join("corpus", "HW2", "CSEP517-HW2-Data", "twt.train.json")
testDataPath = os.path.join("corpus", "HW2", "CSEP517-HW2-Data", "twt.test.json")
bonusDataPath = os.path.join("corpus", "HW2", "CSEP517-HW2-Data", "twt.bonus.json")

wordCount = Counter()
unkWords = set()
unkCandidates = set()
unk_symbol = "<unk>"
unk_threshold=1


# Design decisions
# 1. Should I UNK only if the words appear less than x times, or if the word-tag combination occurs less than x times ?
#   1.a if I unk only the words, do I let the tag be as it is ?
#   1.b if I unk the word-tag combination, do I add a new tag type ? Now that I think about it, doing so does not make a lot of sense because 

def __loadRawData__(path):
    logger.info("Loading data from: %s" % os.path.abspath(path))
    data = []
    with open(path) as f:
        for line in f.readlines():
            data.append(np.asarray(json.loads(line), dtype=object))
    return np.asarray(data, dtype=object)


def unkTrain(line, max_unks):

    for i, word in enumerate(line[:,0]):
        if word in unkCandidates and wordCount[unk_symbol] < max_unks:
            line[i,0] = unk_symbol
            wordCount[unk_symbol]+=1
            unkWords.add(word)
    return line

def unk(line):
    for i, word in enumerate(line[:,0]):
        if word not in wordCount.keys() or word in unkWords:
            line[i,0] = unk_symbol
    return line


def find_unks(corpus):

    for i in range(len(corpus)):
        tokens = corpus[i][:,0]
        for word in tokens:
            wordCount[word] += 1

    for word in wordCount.keys():
        if wordCount[word] <= unk_threshold:
            unkCandidates.add(word)

    logger.debug("unk unkCandidates count : %d", len(unkCandidates))
    logger.debug("unk unkCandidates : %s", unkCandidates)


def loadRawTrainData(unk_percentage = 0.01):
    trainCorpus = __loadRawData__(trainDataPath)
    find_unks(trainCorpus)
    max_unks = unk_percentage * sum(wordCount.values())
    logger.debug("Max unk words in train: %d", max_unks)
    for line in trainCorpus:
        unkTrain(line, max_unks)
    logger.debug(trainCorpus)
    return trainCorpus

def loadRawDevData():
    devCorpus = __loadRawData__(devDataPath)
    for line in devCorpus:
        unk(line)
    return devCorpus


def loadRawTestData():
    testCorpus = __loadRawData__(testDataPath)
    for line in testCorpus:
        unk(line)
    return testCorpus

def loadRawBonusData():
    bonusCorpus = __loadRawData__(bonusDataPath)
    for line in bonusCorpus:
        unk(line)
    return bonusCorpus