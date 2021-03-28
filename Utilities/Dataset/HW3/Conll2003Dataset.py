import os
import logging
from collections import Counter
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import json
from datetime import datetime

format = "%(asctime)s: %(message)s"
logfileName = os.path.join("logs", "HW3", datetime.now().strftime('log_%H_%M_%d_%m_%Y.log') )
logging.basicConfig(filename=logfileName, filemode='a', format=format, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger('__HW3__')

devDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.dev")
devSmallDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.dev.small")
trainDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.train")
trainSmallDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.train.small")
testDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.test")
testSmallDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.test.small")

wordCount = Counter()
unkWords = set()
unkCandidates = set()
unk_symbol = "<unk>"
unk_threshold=1

def __loadRawData__(path):
    logger.info("Loading data from: %s" % os.path.abspath(path))
    data = []
    with open(path) as f:
        statement = []
        for line in f.readlines():
            if len(line) == 1 and line == "\n":
                statement = np.asarray(statement, dtype=object)

                ## Ignore Docstart lines

                # if statement[0,0] != '-DOCSTART-':
                data.append(statement)
                statement = []
            else: 
                statement.append(line.split())

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

def loadRawTrainDataSmall(unk_percentage = 0.001):
    trainCorpus = __loadRawData__(trainSmallDataPath)
    find_unks(trainCorpus)
    max_unks = unk_percentage * sum(wordCount.values())
    logger.debug("Max unk words in train: %d", max_unks)
    for line in trainCorpus:
        unkTrain(line, max_unks)
    logger.debug(trainCorpus)
    return trainCorpus

def loadRawDevDataSmall():
    devCorpus = __loadRawData__(devSmallDataPath)
    for line in devCorpus:
        unk(line)
    return devCorpus


def loadRawTestDataSmall():
    testCorpus = __loadRawData__(testSmallDataPath)
    for line in testCorpus:
        unk(line)
    return testCorpus

def loadRawTrainData(unk_percentage = 0.001):
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