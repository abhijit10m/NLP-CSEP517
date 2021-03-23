import Utilities.Dataset.HW1.BrownDatasetSmall as BrownDatasetSmall
import Utilities.Dataset.HW1.BrownDatasetFull as BrownDatasetFull
import Utilities.LanguageModel.NGram as NGram
import itertools
from datetime import datetime
import os
import logging

format = "%(asctime)s: %(message)s"
logfileName = os.path.join("logs", "HW1", datetime.now().strftime('log_%H_%M_%d_%m_%Y.log') )
logging.basicConfig(filename=logfileName, filemode='a', format=format, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger('__main__')

trainCorpusSmall = BrownDatasetSmall.loadRawTrainData()
devCorpusSmall = BrownDatasetSmall.loadRawDevData()
testCorpusSmall = BrownDatasetSmall.loadRawTestData()
trainCorpusFull = BrownDatasetFull.loadRawTrainData()
devCorpusFull = BrownDatasetFull.loadRawDevData()
testCorpusFull = BrownDatasetFull.loadRawTestData()

TEST=False

if TEST:

    ngram = NGram.NGram(1)
    ngram.train(trainCorpusSmall)
    ngram.perplexityScore(devCorpusSmall,n=1)

    ngram = NGram.NGram(2)
    ngram.train(trainCorpusSmall)
    ngram.perplexityScore(devCorpusSmall,n=2)

    ngram = NGram.NGram(3)
    ngram.train(trainCorpusSmall)

    ngram.perplexityScore(devCorpusSmall,n=3)

    ### Part 2 - Add K smoothing


    k = 0.001

    ngram = NGram.NGram(1)
    ngram.train(trainCorpusSmall)
    ngram.perplexityScore(devCorpusSmall,n=1)

    ngram = NGram.NGram(2)
    ngram.train(trainCorpusSmall)
    ngram.perplexityScore(devCorpusSmall,n=2)


    ngram = NGram.NGram(3)
    ngram.train(trainCorpusSmall)                        
    ngram.perplexityScore(devCorpusSmall,n=3)

    K=[ 0.01, 0.1, 1, 2, 5, 10, 20]

    for k in K: 
        ngram = NGram.NGram(1)
        ngram.train(trainCorpusSmall)
        ngram.perplexityScore(devCorpusSmall,n=1)

        ngram = NGram.NGram(2)
        ngram.train(trainCorpusSmall)
        ngram.perplexityScore(devCorpusSmall,n=2)

    k = 1
    ngram = NGram.NGram(max_n=3)
    ngram.train(trainCorpusSmall)
    ngram.perplexityScore(devCorpusSmall, n=3)

    K= [0.01, 0.1, 1, 2, 5, 10]
    L1 = [0.05, 0.1, 0.15, 0.2, 0.25]
    L2 = [0.1, 0.2, 0.25, 0.4, 0.25]
    a = [K, L1, L2]
    params = list(itertools.product(*a))

    print(params)

DEVELOP=False
if DEVELOP:
    #########################################
    # Actual Assignment questions begin here
    #########################################

    ngramSmall = NGram.NGram(max_n=3)
    ngramFull = NGram.NGram(max_n=3)

    ngramSmall.train(trainCorpusSmall)
    # ngramFull.train(trainCorpusFull)

    ## Part 1

    logger.info("perplexity on trainCorpusSmall with unigrams %f",ngramSmall.perplexityScore(trainCorpusSmall, n=1, K=0))
    logger.info("perplexity on devCorpusSmall with unigrams %f", ngramSmall.perplexityScore(devCorpusSmall, n=1, K=0))
    logger.info("perplexity on testCorpusSmall with unigrams %f", ngramSmall.perplexityScore(testCorpusSmall, n=1, K=0))

    # perplexity on trainCorpusSmall with unigrams 1001.958852
    # perplexity on devCorpusSmall with unigrams 837.592034
    # perplexity on testCorpusSmall with unigrams 837.501113

    logger.info("perplexity on trainCorpusSmall with bigrams %f", ngramSmall.perplexityScore(trainCorpusSmall, n=2, K=0))
    # perplexity on trainCorpusSmall with bigrams 33.240650
    logger.info("perplexity on devCorpusSmall with bigrams %f", ngramSmall.perplexityScore(devCorpusSmall, n=2, K=0))
    # perplexity on devCorpusSmall with bigrams inf
    logger.info("perplexity on testCorpusSmall with bigrams %f", ngramSmall.perplexityScore(testCorpusSmall, n=2, K=0))
    # perplexity on testCorpusSmall with bigrams inf

    logger.info("perplexity on trainCorpusSmall with trigrams %f", ngramSmall.perplexityScore(trainCorpusSmall, n=3, K=0))
    # perplexity on trainCorpusSmall with trigrams 3.959869

    logger.info("perplexity on devCorpusSmall with trigrams %f", ngramSmall.perplexityScore(devCorpusSmall, n=3, K=0))
    # perplexity on devCorpusSmall with trigrams inf

    logger.info("perplexity on testCorpusSmall with trigrams %f", ngramSmall.perplexityScore(testCorpusSmall, n=3, K=0))
    #perplexity on testCorpusSmall with trigrams inf

    ## Part 2

    for k in [10, 1, 0.1, 0.01, 0.001]:

        logger.info("perplexity on trainCorpusSmall with trigrams with add K [K=%f] smoothing %f", k, ngramSmall.perplexityScore(trainCorpusSmall, n=3, K=k))
        logger.info("perplexity on devCorpusSmall with trigrams with add K [K=%f] smoothing %f", k, ngramSmall.perplexityScore(devCorpusSmall, n=3, K=k))

# 23:39:35: perplexity on trainCorpusSmall with trigrams with add K [K=10.000000] smoothing 27822.358529
# 23:39:35: perplexity on devCorpusSmall with trigrams with add K [K=10.000000] smoothing 29669.699039
# 23:39:37: perplexity on trainCorpusSmall with trigrams with add K [K=1.000000] smoothing 11076.527281
# 23:39:37: perplexity on devCorpusSmall with trigrams with add K [K=1.000000] smoothing 17003.066190
# 23:39:39: perplexity on trainCorpusSmall with trigrams with add K [K=0.100000] smoothing 2217.251929
# 23:39:39: perplexity on devCorpusSmall with trigrams with add K [K=0.100000] smoothing 9168.199637
# 23:39:41: perplexity on trainCorpusSmall with trigrams with add K [K=0.010000] smoothing 380.062887
# 23:39:41: perplexity on devCorpusSmall with trigrams with add K [K=0.010000] smoothing 5601.955443
# 23:39:43: perplexity on trainCorpusSmall with trigrams with add K [K=0.001000] smoothing 72.616991
# 23:39:43: perplexity on devCorpusSmall with trigrams with add K [K=0.001000] smoothing 3975.834031

    ## Part 3

    ## Part 4.2

    params = [
        (0.01, 0.2, 0.4, 0.4),
        (0.001, 0.1, 0.4, 0.5),
        (0.0005, 0.05, 0.3, 0.65),
        (0.001, 0.1, 0.3, 0.6),
        (0.0051, 0.0, 0.1, 0.9)
    ]


    for param in params:
        logger.info("perplexity on trainCorpusSmall with trigrams with Linear Interpolation [K=%f, L1=%f, L2=%f, L3=%f] %f", param[0], param[3],param[2],param[1], ngramSmall.perplexityScore(trainCorpusSmall, n=3, K=param[0], useLinearInterpolation=True, lambdas=param[1:]))
        logger.info("perplexity on devCorpusSmall with trigrams with Linear Interpolation [K=%f, L1=%f, L2=%f, L3=%f] %f", param[0], param[3],param[2],param[1], ngramSmall.perplexityScore(devCorpusSmall, n=3, K=param[0], useLinearInterpolation=True, lambdas=param[1:]))

# 23:42:50: perplexity on trainCorpusSmall with trigrams with Linear Interpolation [K=0.010000, L1=0.400000, L2=0.400000, L3=0.200000] 131.408409
# 23:42:51: perplexity on devCorpusSmall with trigrams with Linear Interpolation [K=0.010000, L1=0.400000, L2=0.400000, L3=0.200000] 495.464993
# 23:42:55: perplexity on trainCorpusSmall with trigrams with Linear Interpolation [K=0.001000, L1=0.500000, L2=0.400000, L3=0.100000] 43.808302
# 23:42:56: perplexity on devCorpusSmall with trigrams with Linear Interpolation [K=0.001000, L1=0.500000, L2=0.400000, L3=0.100000] 375.630557
# 23:43:00: perplexity on trainCorpusSmall with trigrams with Linear Interpolation [K=0.000500, L1=0.650000, L2=0.300000, L3=0.050000] 33.027767
# 23:43:01: perplexity on devCorpusSmall with trigrams with Linear Interpolation [K=0.000500, L1=0.650000, L2=0.300000, L3=0.050000] 399.608084
# 23:43:05: perplexity on trainCorpusSmall with trigrams with Linear Interpolation [K=0.001000, L1=0.600000, L2=0.300000, L3=0.100000] 46.759688
# 23:43:06: perplexity on devCorpusSmall with trigrams with Linear Interpolation [K=0.001000, L1=0.600000, L2=0.300000, L3=0.100000] 425.560746
# 23:43:10: perplexity on trainCorpusSmall with trigrams with Linear Interpolation [K=0.005100, L1=0.900000, L2=0.100000, L3=0.000000] 135.389468
# 23:43:11: perplexity on devCorpusSmall with trigrams with Linear Interpolation [K=0.005100, L1=0.900000, L2=0.100000, L3=0.000000] 812.910972


    #### Here evaluate on full test set for best value

    # Best Values = [K=0.001000, L1=0.500000, L2=0.400000, L3=0.100000]





BRING_IT_TOGETHER = True
if BRING_IT_TOGETHER:

    ngramSmall = NGram.NGram(max_n=3)
    ngramSmall.train(trainCorpusSmall)
    ngramFull = NGram.NGram(max_n=3)
    ngramSmall.train(trainCorpusFull)


    # Q4.1

    logger.info("perplexity on trainCorpusFull with unigrams %f",ngramSmall.perplexityScore(trainCorpusFull, n=1, K=0))
    logger.info("perplexity on devCorpusFull with unigrams %f", ngramSmall.perplexityScore(devCorpusFull, n=1, K=0))
    logger.info("perplexity on testCorpusFull with unigrams %f", ngramSmall.perplexityScore(testCorpusFull, n=1, K=0))

    # 23:49:20: perplexity on trainCorpusFull with unigrams 1033.365181
    # 23:49:21: perplexity on devCorpusFull with unigrams 908.892538
    # 23:49:21: perplexity on testCorpusFull with unigrams 908.611301

    logger.info("perplexity on trainCorpusFull with bigrams %f", ngramSmall.perplexityScore(trainCorpusFull, n=2, K=0))
    logger.info("perplexity on devCorpusFull with bigrams %f", ngramSmall.perplexityScore(devCorpusFull, n=2, K=0))
    logger.info("perplexity on testCorpusFull with bigrams %f", ngramSmall.perplexityScore(testCorpusFull, n=2, K=0))

    # 23:49:24: perplexity on trainCorpusFull with bigrams 41.585063
    # 23:49:24: perplexity on devCorpusFull with bigrams inf
    # 23:49:24: perplexity on testCorpusFull with bigrams inf

    logger.info("perplexity on trainCorpusFull with trigrams %f", ngramSmall.perplexityScore(trainCorpusFull, n=3, K=0))
    logger.info("perplexity on devCorpusFull with trigrams %f", ngramSmall.perplexityScore(devCorpusFull, n=3, K=0))
    logger.info("perplexity on testCorpusFull with trigrams %f", ngramSmall.perplexityScore(testCorpusFull, n=3, K=0))

    # 23:49:27: perplexity on trainCorpusFull with trigrams 4.916969
    # 23:49:28: perplexity on devCorpusFull with trigrams inf
    # 23:49:28: perplexity on testCorpusFull with trigrams inf

    ## Part 2

    for k in [10, 1, 0.1, 0.01, 0.001]:

        logger.info("perplexity on trainCorpusFull with trigrams with add K [K=%f] smoothing %f", k, ngramSmall.perplexityScore(trainCorpusFull, n=3, K=k))
        logger.info("perplexity on devCorpusFull with trigrams with add K [K=%f] smoothing %f", k, ngramSmall.perplexityScore(devCorpusFull, n=3, K=k))

    # 23:49:31: perplexity on trainCorpusFull with trigrams with add K [K=10.000000] smoothing 34725.034217
    # 23:49:31: perplexity on devCorpusFull with trigrams with add K [K=10.000000] smoothing 38111.692445
    # 23:49:35: perplexity on trainCorpusFull with trigrams with add K [K=1.000000] smoothing 11776.535027
    # 23:49:35: perplexity on devCorpusFull with trigrams with add K [K=1.000000] smoothing 19867.568063
    # 23:49:38: perplexity on trainCorpusFull with trigrams with add K [K=0.100000] smoothing 2231.352480
    # 23:49:39: perplexity on devCorpusFull with trigrams with add K [K=0.100000] smoothing 10029.610863
    # 23:49:42: perplexity on trainCorpusFull with trigrams with add K [K=0.010000] smoothing 391.491873
    # 23:49:42: perplexity on devCorpusFull with trigrams with add K [K=0.010000] smoothing 5749.768621
    # 23:49:46: perplexity on trainCorpusFull with trigrams with add K [K=0.001000] smoothing 78.056750
    # 23:49:46: perplexity on devCorpusFull with trigrams with add K [K=0.001000] smoothing 3855.158297


    # Q2
    logger.info("perplexity on testCorpusSmall with trigrams with add K [K=%f] smoothing %f", 0.001, ngramSmall.perplexityScore(testCorpusSmall, n=3, K=0.001))
    # 23:45:46: perplexity on testCorpusSmall with trigrams with add K [K=0.001000] smoothing 3985.191197
    # Q4.2
    logger.info("perplexity on testCorpusSmall with trigrams with Linear Interpolation [K=%f, L1=%f, L2=%f, L3=%f] %f", 0.001000 , 0.5, 0.4, 0.1, ngramSmall.perplexityScore(testCorpusSmall, n=3, K=0.001, useLinearInterpolation=True, lambdas=[0.1,0.4,0.5]))    
    # 23:45:47: perplexity on testCorpusSmall with trigrams with Linear Interpolation [K=0.001000, L1=0.500000, L2=0.400000, L3=0.100000] 374.952455