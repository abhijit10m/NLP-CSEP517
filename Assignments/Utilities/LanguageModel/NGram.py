from collections import Counter, OrderedDict
import math
import logging
from datetime import datetime
import os
from joblib import Parallel, delayed
import multiprocessing.dummy as mp


format = "%(asctime)s: %(message)s"
logfileName = os.path.join("logs", "HW1", str(os.getpid())+"_"+datetime.now().strftime('log_%H_%M_%d_%m_%Y.log') )
logging.basicConfig(filename=logfileName, filemode='a', format=format, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger('__main__')

def curateLine(line, start, end):
    line[-1] = line[-1].rstrip()
    logger.debug("start: %s, end: %s, line_stripped: [%s]", start, end, line[-1])
    return start + line + end

def getNGrams(words, n):
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    logger.debug("ngrams: %s", ngrams)
    return ngrams

class NGram(object):
    def __init__(self, max_n, start_symbol="<s>", end_symbol="<e>"):
        self.max_n = max_n
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.wordCount = Counter()
        self.ngramCount = [Counter() for i in range(self.max_n)]
        self.ngrams = [ [] for i in range(self.max_n)]
        self.totalWords = 0
        self.ngram_sums = [0 for i in range(self.max_n)]
        self.unique_ngram_counts = [0 for i in range(self.max_n)]

    def train(self, corpus):
        start = [self.start_symbol for i in range(self.max_n)]
        end = [self.end_symbol for i in range(self.max_n)]

        curated_corpus = []
        for i in range(len(corpus)):
            curated_corpus.append(curateLine(corpus[i], start, end))
            tokens = curated_corpus[i]
            for word in tokens:
                self.wordCount[word] += 1

        for i in range(len(curated_corpus)):
            words = curated_corpus[i]

            logger.debug("words: %s", words)

            for j in range(1, self.max_n+1):
                ngrams = getNGrams(words, j)
                for ngram in ngrams:
                    self.ngramCount[j-1][ngram] += 1

        for i in range(self.max_n):
            self.ngrams[i] = sorted(self.ngramCount[i].keys())
            logger.info("%d ngrams count %d", i+1, len(self.ngrams[i]))
            self.ngram_sums[i] = sum(self.ngramCount[i].values())
            self.unique_ngram_counts[i] = len(self.ngramCount[i].keys())
        self.totalWords = sum(self.ngramCount[0].values())
        self.totalUniqueWords= len(self.wordCount.keys())

    def testProbabilityDistribution(self, K=1):
        for i in reversed(range(self.max_n)):
            for nGram in self.ngrams[i]:
                prefix = nGram.rpartition(" ")[0]
                matchingNGrams = [ matching_ngram for matching_ngram in self.ngrams[i] if matching_ngram.startswith(prefix)]
                c_ngram = 0
                for matchingNGram in matchingNGrams:

                    c_ngram += self.ngramCount[i][matchingNGram] + K

                c = sum([self.ngramCount[i][m] for m in matchingNGrams]) + (K * len(matchingNGrams))
                logger.debug("ngram: %s c: %f c_ngram: %f", nGram, c, c_ngram)
                assert c_ngram == c

        logger.debug("__stub__ testProbabilityDistribution")


    def ngramProbability(self, ngram, n, K, lambdas):

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

    def lineProbability(self, words, n, K, useLinearInterpolation, lambdas):
        evalNGrams = getNGrams(words, n)
        p = 0
        for nGram in evalNGrams:
            mle = self.ngramProbability(nGram, n, K, lambdas)

            if useLinearInterpolation:
                for i in range(1, n):
                    suffix = nGram.partition(" ")[-1]
                    mle +=  self.ngramProbability(suffix, n - i, K, lambdas)

            if mle == 0:
                return float('inf')

            p += math.log2(mle)
        return p

    def parallelPerplexityScore(self, words_curated, counter, n, K, useLinearInterpolation, lambdas):

        l = self.lineProbability(words_curated, n, K, useLinearInterpolation, lambdas)
        logger.debug("Perplexity Counter: %d, l: %f, eval_line_curated: %s", counter, l, words_curated)
        return l


    def parallelCurateCorpus(self, eval_line, counter, n, start, end):
        words = curateLine(eval_line, start, end)
        logger.debug("Curate counter: %d, words: %s", counter, words)
        return words

    def perplexityScore(self, eval_corpus, n, K=1, useLinearInterpolation=False, lambdas=[]):
        start = [self.start_symbol for i in range(n)]
        end = [self.end_symbol for i in range(n)]

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

        for words in eval_corpus_curated:
            M+=len(words)

        logger.debug("M total: %d", M)

        params = [[words, counter, n, K, useLinearInterpolation, lambdas] for (counter, words)  in enumerate(eval_corpus_curated)]
        
        with mp.Pool() as p:
            scores = p.starmap(self.parallelPerplexityScore, params)
        p.close()
        p.join()

        l = sum(scores)

        if l == float('inf'):
            return float('inf')
        l *= (1 / M)
        perplexity = math.pow(2, -1 * l)
        logger.debug("perplexity %f", perplexity)
        return perplexity