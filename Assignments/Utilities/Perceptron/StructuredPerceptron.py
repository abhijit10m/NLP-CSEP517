import os
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
import multiprocessing.dummy as mp

format = "%(asctime)s: %(message)s"
logfileName = os.path.join("logs", "HW3", str(os.getpid())+"_"+datetime.now().strftime('progress_log_%H_%M_%d_%m_%Y.log') )
logging.basicConfig(filename=logfileName, filemode='a', format=format, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger("__HW3__")

class StructuredPerceptron(object):
    def __init__(self, featureCounts, featurizer):
        self.featureCounts = featureCounts
        self.featurizer = featurizer
        self.a = [[defaultdict(int) for n in self.featurizer.named_entity_recognition_tags.keys()] for featureCount in featureCounts]

    def updateWeights(self, yPredicted, y, x):
        logger.debug("y: %s, yP: %s", y, yPredicted)
        for l in range(len(x)): # once per word
            for i in range(len(x[l])): # once per feature_set for a word

                y_ner_index = self.featurizer.named_entity_recognition_tags[yPredicted[l]]
                x_ner_index = self.featurizer.named_entity_recognition_tags[y[l]]
                
                logger.debug("subtracting from l, %d, i, %d, y_ner_index, %d", l, i, y_ner_index)

                for feature in x[l][i]: # once per feature in a feature set
                    self.a[i][x_ner_index][feature] += 1
                    self.a[i][y_ner_index][feature] -= 1

        logger.debug("updated a: %s", self.a)


    # Rows: named_entity_recognition_tags: {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'I-LOC': 3, 'I-MISC': 4, 'I-ORG': 5, 'I-PER': 6, 'O': 7}

    def viterbi(self, x, nT):
        T = len(x)
        N = len(self.featurizer.named_entity_recognition_tags.keys())

        logger.debug("T: %d, N: %d, N: %s", T, N, self.featurizer.named_entity_recognition_tags.keys())
        viterbi = np.full(shape=(T + 4, N), fill_value=0, dtype=float)
        backpointer = np.full(shape=(T + 4, N), fill_value=None, dtype=object)

        end_tag_index = self.featurizer.named_entity_recognition_tags['O']
        start_tag_index = self.featurizer.named_entity_recognition_tags['O']

        viterbi[0][start_tag_index] = 1
        viterbi[1][start_tag_index] = 1
        backpointer[0,start_tag_index] = (0,start_tag_index)
        backpointer[1,start_tag_index] = (0,start_tag_index)
        backpointer[T + 3,end_tag_index] = (T + 2,end_tag_index)

        for t in range(2, T+2):
            for s in range(N):
                scores = [viterbi[t-1,j] for j in range(N)]
                logger.debug("scores t, s, i, j : %d, %d, %s", t, s, scores)
                s_score = 0
                logger.debug("x : %s", x[t-2])
                for i in range(len(x[t-2])):
                    logger.debug("feature_set %d, ner %s, features %s", i, s,  x[t-2][i])
                    for feature in x[t-2][i].keys():
                        s_score += self.a[i][s][feature]/nT * x[t-2][i][feature]
                scores =  [s_score + pre_score for pre_score in scores]
                logger.debug("s_score %0.20f, scores: %s", s_score, scores)
                viterbi[t,s] = max(scores)
                backpointer[t,s] = (t-1,) + (scores.index(viterbi[t,s]),)

        scores = [viterbi[T+1,j] for j in range(N)]
        viterbi[T+2,end_tag_index] = max(scores)
        backpointer[T+2,end_tag_index] = (T + 1,) + (scores.index(max(scores)),)

        logger.debug("viterbi: %s", viterbi)
        logger.debug("backpointer: %s", backpointer)
        y = []

        p = (T+3, end_tag_index)

        for i in reversed(range(T+4)):
            y.append(self.featurizer.named_entity_recognition_tags_index[p[1]])
            p = backpointer[p]

        return list(reversed(y))

    def trainOnce(self, xTrain, yTrain, nt, pbar):
        for (x, y_t) in zip(xTrain, yTrain):
            y = self.viterbi( x, nt)
            y = y[ 2: len(y) - 2]
            self.updateWeights(y, y_t, x)
            logger.debug("train: %d, %s", len(x) ,x)
            logger.debug("predict: %s", y)
            pbar.update(1)

    def train(self, xTrain, yTrain, T=1):

        pbar = tqdm(total=len(xTrain), desc="Training")
        self.train_n = len(xTrain)
        logger.debug(xTrain)

        self.trainOnce(xTrain, yTrain, self.train_n * T, pbar)

        self.train_t = T + 1

        pbar.close()

    def predictOnce(self, x):
        nt = self.train_n * self.train_t
        y = self.viterbi(x, nt)
        self.predict_pbar.update(1)
        logger.debug("predict: %s", x)
        logger.debug("predict: %s", y[2: len(y) - 2])
        return y[2: len(y) - 2]

    def predict(self, corpus):
        self.predict_pbar = tqdm(total=len(corpus), desc="Predicting")

        with mp.Pool(processes=20) as p:
            y = p.map(self.predictOnce, corpus)
        p.close()
        p.join()

        self.predict_pbar.close()
        return y