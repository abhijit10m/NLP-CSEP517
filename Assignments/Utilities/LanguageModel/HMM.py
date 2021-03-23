from collections import Counter, OrderedDict, defaultdict
from itertools import permutations, product
import math
import logging
from datetime import datetime
import os
from joblib import Parallel, delayed
import multiprocessing.dummy as mp
import numpy as np
from tqdm.auto import tqdm

import threading
threadLock = threading.Lock()

class TqdmFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x, end='\n'):
        if len( x.rstrip() ) > 0:
            tqdm.write( x, file=self.file, end=end )

    def flush(self):
        return getattr( self.file, "flush", lambda: None )()


format = "%(asctime)s: %(message)s"
logfileName = os.path.join("logs", "HW2", str(os.getpid())+"_"+datetime.now().strftime('progress_log_%H_%M_%d_%m_%Y.log') )
logging.basicConfig(filename=logfileName, filemode='a', format=format, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def curateLine(line, start, end):
    line = np.insert(line,0,values=start, axis=0)
    line = np.append(line,values=end, axis=0)
    return line

def getNGrams(words, n):
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

class HMM(object):
    def __init__(self, max_n, start_symbol="<s>", end_symbol="<e>"):
        self.max_n = max_n
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.ngramCount = [Counter() for i in range(self.max_n)]
        self.ngrams = [ [] for i in range(self.max_n)]
        self.ngrams_index = [ [] for i in range(self.max_n)]
        self.totalWords = 0
        self.ngram_sums = [0 for i in range(self.max_n)]
        self.unique_ngram_counts = [0 for i in range(self.max_n)]
        self.emissionCounters = defaultdict(lambda: Counter())
        self.startTags = ['' for i in range(self.max_n)]
        self.correct_counter = 0
        self.word_counter = 0

    def train(self, corpus):

        start = [[self.start_symbol, self.start_symbol] for i in range(self.max_n - 1)]
        end = [[self.end_symbol, self.end_symbol] for i in range(self.max_n - 1)]

        curated_corpus = []
        for i in range(len(corpus)):
            curated_corpus.append(curateLine(corpus[i], start, end))
            tokens = curated_corpus[i][:,0]
            tags = curated_corpus[i][:,1]
            for ix, word in enumerate(tokens):
                self.emissionCounters[tags[ix]][word]+=1

        for i in range(len(curated_corpus)):
            tags = curated_corpus[i][:,1]

            logger.debug("tags: %s", tags)

            for j in range(1, self.max_n+1):
                ngrams = getNGrams(tags, j)
                for ngram in ngrams:
                    self.ngramCount[j-1][ngram] += 1

        for i in range(self.max_n):
            self.ngrams[i] = dict(((ngram, index) for (index, ngram) in enumerate(sorted(self.ngramCount[i].keys()))))
            self.ngrams_index[i] = dict(enumerate(sorted(self.ngramCount[i].keys())))
            logger.debug("%d ngrams count %d", i+1, len(self.ngrams[i].keys()))
            self.ngram_sums[i] = sum(self.ngramCount[i].values())
            self.unique_ngram_counts[i] = len(self.ngramCount[i].keys())

        for i in range(1, self.max_n):
            start_tag_array = [self.start_symbol] * i
            logger.debug("%d start_tag_array %s", i+1, start_tag_array)
            start_tag = " ".join(start_tag_array)
            logger.debug("%d start_tag '%s'", i+1, start_tag)
            self.startTags[i] = start_tag
            logger.debug("%d startTags %s", i+1, self.startTags)

        logger.debug("emissionCounters: %s", self.emissionCounters)
        logger.debug("emissionCounters keys: %s", self.emissionCounters.keys())
        logger.debug("ngrams: %s", self.ngrams)
        logger.debug("ngrams_index: %s", self.ngrams_index)
        logger.debug("unique_ngram_counts: %s", self.unique_ngram_counts)
        logger.debug("ngram_counts: %s", self.ngramCount)
        logger.debug("ngrams_index 0: %s", self.ngrams_index[0])



    def ngramProbability(self, ngram, n, K, lambdas, ngram_probabilities):

        p = 0
        c = 0
        c_ngram = self.ngramCount[n-1][ngram] + K

        prefix = ngram.rpartition(" ")[0]

        if n > 1:

            matchingNGramsCounts = self.ngramCount[n-2][prefix]

            if matchingNGramsCounts > 0:
                logger.debug("[%d] matched [%s]", matchingNGramsCounts, ngram)
            else:
                logger.debug("No matches for [%s]", ngram)

            c = matchingNGramsCounts + (K * self.unique_ngram_counts[n-2])
        else:
            c = self.ngram_sums[0] + (K * self.unique_ngram_counts[0])

        if c == 0:
            mle = 0
        else:
            mle = lambdas[n-1] * c_ngram / c

        logger.debug("ngram [%s], mle %f, l: %f", ngram, mle, lambdas[n-1])
        return mle

    def viterbiTriGram(self, eval_line, viterbi, backpointer, ngram_probabilities, emission_probabilities):

        T = len(eval_line)
        N = self.unique_ngram_counts[0]

        backpointer[0,self.ngrams[0][self.start_symbol],self.ngrams[0][self.start_symbol]] = (0,self.ngrams[0][self.start_symbol],self.ngrams[0][self.start_symbol])

        def trigramProbability(w1, w2, w3):
            return ngram_probabilities[2][" ".join([w1,w2,w3])]

        for t in range(1, T-1):
            for u in range(N):
                for v in range(N):
                    probabilities = np.asarray([viterbi[t-1,w,u] + trigramProbability(self.ngrams_index[0][w], self.ngrams_index[0][u], self.ngrams_index[0][v]) + emission_probabilities[self.ngrams_index[0][v]][eval_line[t,0]] for w in range(N) ])
                    viterbi[t,u,v] = np.ndarray.max(probabilities)
                    backpointer[t,u,v] = (t-1,) + np.unravel_index(np.ndarray.argmax(probabilities), (N,N))

                logger.debug("viterbi t : %s", viterbi[t,:])
                logger.debug("backpointer t : %s", backpointer[t,:])

        probabilities = np.asarray([ viterbi[T-2, w, self.ngrams[0][self.end_symbol]] + trigramProbability(self.ngrams_index[0][w], self.end_symbol, self.end_symbol) + emission_probabilities[self.end_symbol][self.end_symbol] for w in range(N) ])
        viterbi[T-1, self.ngrams[0][self.end_symbol], self.ngrams[0][self.end_symbol]] = np.ndarray.max(probabilities)
        backpointer[T-1, self.ngrams[0][self.end_symbol], self.ngrams[0][self.end_symbol]] = (T-2,) + np.unravel_index(np.ndarray.argmax(probabilities), (N,N))

        y = [self.end_symbol]

        logger.debug("Backpointer: %s", backpointer)
        p = (T-1, self.ngrams[0][self.end_symbol], self.ngrams[0][self.end_symbol])

        for i in reversed(range(T-1)):
            y.append(self.ngrams_index[0][p[2]])
            p = backpointer[p]

        logger.debug("Predicted Y: %s, Actual Y: %s", list(reversed(y)), eval_line[:, 1])

        return list(reversed(y))


    def viterbiBiGram(self, eval_line, viterbi, backpointer, ngram_probabilities, emission_probabilities):
        T = len(eval_line)
        N = self.unique_ngram_counts[0]

        backpointer[0, self.ngrams[0][self.start_symbol]] = (0,self.ngrams[0][self.start_symbol])

        def bigramProbability(w1, w2):
            return ngram_probabilities[1][" ".join([w1,w2])]

        for t in range(1, T-1):
            for s in range(N):
                probabilities = np.asarray([viterbi[t-1,j] + bigramProbability(self.ngrams_index[0][j], self.ngrams_index[0][s]) + emission_probabilities[self.ngrams_index[0][s]][eval_line[t,0]] for j in range(N)  ])
                viterbi[t,s] = np.max(probabilities)
                backpointer[t,s] = (t-1,) + (np.argmax(probabilities),)

        probabilities = np.asarray([ viterbi[T-2, j] + bigramProbability(self.ngrams_index[0][j], self.end_symbol) + emission_probabilities[self.end_symbol][self.end_symbol] for j in range(N)  ])
        viterbi[T-1, self.ngrams[0][self.end_symbol]] = np.max(probabilities)
        backpointer[T-1, self.ngrams[0][self.end_symbol]] = (T-2,) + (np.argmax(probabilities),)

        y = []

        logger.debug("Backpointer: %s", backpointer)
        p = (T-1, self.ngrams[0][self.end_symbol])

        logger.debug("Predicted Y: %s, Actual Y: %s", y, eval_line[:, 1])

        for i in reversed(range(T)):
            y.append(self.ngrams_index[0][p[1]])
            p = backpointer[p]
            logger.debug("Predicted Y: %s, Actual Y: %s", y, eval_line[:, 1])

        logger.debug("Predicted Y: %s, Actual Y: %s", list(reversed(y)), eval_line[:, 1])

        return list(reversed(y))

    def getMostLikelyTagSequence(self, eval_line, ngram_probabilities, n, K, pbar):
        T = len(eval_line)
        N = self.unique_ngram_counts[0]

        shape = [T] + [N for i in range(n-1)]
        logger.debug("shape: %s, eval line: %s", shape, eval_line)

        viterbi = np.full(shape=shape, fill_value=float('-inf'), dtype=float)
        backpointer = np.full(shape=shape, fill_value=None, dtype=object)

        emission_probabilities = dict()

        words = set()
        for tag in self.emissionCounters.keys():
            words.update(eval_line[:,0])
            emission_probabilities[tag] = dict()

        logger.debug("words: %s", words)

        for tag in self.emissionCounters.keys():
            total_tag_emissions = sum(self.emissionCounters[tag].values())
            for word in words:
                logger.debug("tag : %s word :%s, emission count, %d, total emission %d", tag, word,  self.emissionCounters[tag][word], total_tag_emissions)
                emission_probabilities[tag][word] = math.log2( (self.emissionCounters[tag][word] + K) / (total_tag_emissions + (K * self.unique_ngram_counts[0])) )


        logger.debug("emission_probabilities %s", emission_probabilities)
        for t in range(n-1):
            viterbi[tuple([t] + [self.ngrams[0][self.startTags[1]] for i in range(n-1) ])] = 0

        if n==2:
            y = self.viterbiBiGram(eval_line, viterbi, backpointer, ngram_probabilities, emission_probabilities)
        elif n==3:
            y = self.viterbiTriGram(eval_line, viterbi, backpointer, ngram_probabilities, emission_probabilities)

        logger.debug("viterbi: %s", viterbi)
        logger.debug("Predicted Y: %s, Actual Y: %s", y, eval_line[:, 1])
        
        assert len(y) == len(eval_line[:, 1])

        accurate_tags = [1 if y[i] == eval_line[i,1] else 0 for i in range(len(eval_line))]
        total_words = len(eval_line) 

        with threadLock:
            self.correct_counter += sum(accurate_tags) -  2*(n-1)
            self.word_counter += total_words - 2*(n-1)
            logger.info("running accuracy: %0.20f, accurate %s, total %d, Predicted Y: %s, Actual Y: %s", self.correct_counter/self.word_counter, accurate_tags, total_words, y, eval_line[:, 1])
        pbar.update(1)

        return y

    def parallelCurateCorpus(self, eval_line, counter, n, start, end):
        eval_line = curateLine(eval_line, start, end)
        logger.debug("Curate counter: %d, eval_line: %s", counter, eval_line)
        return eval_line

    def predict(self, eval_corpus, n, K=1, lambdas=[], useLinearInterpolation=True,progressbar=0):
        start = [[self.start_symbol, self.start_symbol] for i in range(self.max_n - 1)]
        end = [[self.end_symbol, self.end_symbol] for i in range(self.max_n - 1)]

        if n > self.max_n:
            raise "n [{}] must be lower than max_n [{}]".format(n, self.max_n)

        l = 0
        M = 0

        if useLinearInterpolation==False:
            lambdas = [0 for i in range(n)]
            lambdas[n-1] = 1
        else:
            if sum(lambdas) != 1:
                raise "L must sum to 1"

        params = [[eval_line, counter, n, start, end] for (counter, eval_line)  in enumerate(eval_corpus)]
        with mp.Pool(100) as p:
            eval_corpus_curated = p.starmap(self.parallelCurateCorpus, params)
        p.close()
        p.join()

        logger.debug("eval_corpus_curated %s", eval_corpus_curated)

        ngram_probabilities = [ dict() for nx in range(n) ]

        for i in range(n):
            for s in self.ngrams[0].keys():
                all_possible_ngrams = [" ".join(ngram) for ngram in product(list(self.ngrams[0].keys()), repeat=i+1)]
                assert len(all_possible_ngrams) == len(self.ngrams[0].keys()) ** (i+1)
                for ngram in all_possible_ngrams:
                    ngram_probabilities[i][ngram] = self.ngramProbability(ngram, i+1 , K, lambdas, ngram_probabilities)

        logger.debug("ngram probabilities = %s", ngram_probabilities)

        ngram_log_probabilities = [ dict() for nx in range(n) ]

        for i in reversed(range(n)):
            for s in self.ngrams[0].keys():
                all_possible_ngrams = set([" ".join(ngram) for ngram in product(list(self.ngrams[0].keys()), repeat=i+1)])
                for ngram in all_possible_ngrams:
                    words = []
                    for j in reversed(range(i+1)):
                        words.append(" ".join(ngram.split(" ")[j:]))
                    p = [ngram_probabilities[k][word] for (k,word) in enumerate(words)]
                    logger.debug("ngram: %s is made up for ngrams: %s, with linear interpolation p = %s, sum = %0.10f ", ngram, words, p, sum(p))
                    ngram_log_probabilities[i][ngram] = math.log2(sum(p))


        logger.debug("ngram probabilities = %s", ngram_log_probabilities)

        
        pbar = tqdm(total=len(eval_corpus_curated), position=progressbar, desc=" param {}".format(progressbar))

        params = [[words, ngram_log_probabilities, n, K, pbar] for counter, words in enumerate(eval_corpus_curated)]

        self.correct_counter = 0
        self.word_counter = 0

        with mp.Pool(4) as p:
            predictions = p.starmap(self.getMostLikelyTagSequence, params)
        p.close()
        p.join()
        pbar.close()

        accuracy = self.correct_counter/self.word_counter
        logger.info("i: %d, accuracy: %0.10f", i, accuracy)
        return accuracy