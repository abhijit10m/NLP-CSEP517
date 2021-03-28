from Utilities.Featurizer.ConllFeaturizer import ConllFeaturizer
from Utilities.Perceptron.StructuredPerceptron import StructuredPerceptron

import Utilities.Dataset.HW3.Conll2003Dataset as Conll2003Dataset
import itertools
from datetime import datetime
import os
import logging
import sys
import numpy as np
from tqdm.auto import tqdm
import subprocess

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
np.set_printoptions(threshold=sys.maxsize)

format = "%(asctime)s: %(message)s"
logfileName = os.path.join("logs", "HW3", datetime.now().strftime('log_%H_%M_%d_%m_%Y.log') )
logging.basicConfig(filename=logfileName, filemode='a', format=format, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger('__HW3__')

devOutDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.dev.out")
devSmallOutDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.dev.small.out")
trainOutDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.train.out")
trainSmallOutDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.train.small.out")
testOutDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.test.out")
testSmallOutDataPath = os.path.join("corpus", "HW3", "conll03_ner", "eng.test.small.out")

conllevalScript = os.path.join("corpus", "HW3", "conll03_ner", "conlleval.txt")


def zipResults(input,output):
    x_return = []
    for t in range(len(input)):
        x_return.append(np.column_stack((input[t], output[t])))
    return x_return

def outResults(filePath, out):

    pipe = subprocess.Popen(["perl", conllevalScript], stdin=subprocess.PIPE)
    with open(filePath, 'w') as filehandle:
        for sample in out:
            for pred in sample:

                line = '%s\n' % " ".join(pred)
                filehandle.write(line)
                pipe.stdin.write(line.encode())
            filehandle.write('\n')
            pipe.stdin.write('\n'.encode())

    pipe.stdin.close()



if __name__ == "__main__":

    trainData = Conll2003Dataset.loadRawTrainData()
    devData = Conll2003Dataset.loadRawDevData()
    testData = Conll2003Dataset.loadRawTestData()
    trainDataSmall = Conll2003Dataset.loadRawTrainDataSmall()
    devDataSmall = Conll2003Dataset.loadRawDevDataSmall()
    testDataSmall = Conll2003Dataset.loadRawTestDataSmall()

    featurizer = ConllFeaturizer(np.concatenate((trainData, devData, testData, trainDataSmall, devDataSmall, testDataSmall)))

    DEVEL = False
    if DEVEL:

        DEVEL_DATASET=False
        if DEVEL_DATASET:
            logger.info("trainData 0 %s", trainData[0])
            logger.info("trainData 0 words %s", trainData[0][:,0])
            logger.info("trainData 0 POS %s", trainData[0][:,1])
            logger.info("trainData 0 context %s", trainData[0][:,2])
            logger.info("trainData 0 NER %s", trainData[0][:,3])

        DEVEL_FEATURIZER=False
        if DEVEL_FEATURIZER:
            featurizer = ConllFeaturizer(trainData)
            x, featureCountsPerWord = featurizer.featurizeTrain(trainData)
            logger.debug("total featuresPerWord = %s", featureCountsPerWord)
            for ex in x: 
                logger.debug("Features: %s", ex[:,:4])

        DEVEL_ALGO=False
        if DEVEL_ALGO:
            featurizer.word_featurizers=[
                (featurizer.w_is_number_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
                (featurizer.w_is_capitalized_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
                (featurizer.w_is_split_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
                (featurizer.w_1_gram_tag_featurizer, len(featurizer.tag_ngrams[0])),
                (featurizer.w_2_gram_tag_featurizer, len(featurizer.tag_ngrams[1])),
                (featurizer.w_3_gram_tag_featurizer, len(featurizer.tag_ngrams[2])),
                (featurizer.w_2_gram_forward_tag_featurizer, len(featurizer.tag_ngrams[1])),
                (featurizer.w_3_gram_forward_tag_featurizer, len(featurizer.tag_ngrams[2])),
                (featurizer.w_3_gram_around_tag_featurizer, len(featurizer.tag_ngrams[2])),
                (featurizer.w_1_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[0])),
                (featurizer.w_2_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[1])),
                (featurizer.w_3_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
                (featurizer.w_2_gram_forward_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[1])),
                (featurizer.w_3_gram_forward_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
                (featurizer.w_3_gram_around_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
                (featurizer.w_synctactic_chunk_word_featurizer, len(featurizer.syntactic_chunk_word)),
                (featurizer.w_syntactic_chunk_tag_featurizer, len(featurizer.syntactic_chunk_tag)),
                (featurizer.w_word_tag_featurizer, len(featurizer.word_tag)),
        ]   
            xTrain, featureCountsPerWordTrain = featurizer.featurizeTrain(trainData)
            yTrain = [t[:,3] for t in trainData]

            xDev, featureCountsPerWordDev = featurizer.featurize(devData)
            xTest, featureCountsPerWordTest = featurizer.featurize(testData)
            
            perceptron = StructuredPerceptron(featureCountsPerWordTrain, featurizer)
            perceptron.train(xTrain, yTrain, T=1)
            yTrainPredicted = perceptron.predict(xTrain)

            trainData = zipResults(trainData, yTrainPredicted)
            outResults(trainOutDataPath, trainData)

            yDev = perceptron.predict(xDev)

            devData = zipResults(devData, yDev)
            outResults(devOutDataPath, devData)

            yTest = perceptron.predict(xTest)

            testData = zipResults(testData, yTest)
            outResults(testOutDataPath, testData)


    featurizer.word_featurizers=[]

    ZEROGRAM=False
    if ZEROGRAM:
        featurizer.word_featurizers=[]

    W_SYNCTACTIC_CHUNK_WORD_FEATURIZER=False
    if W_SYNCTACTIC_CHUNK_WORD_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_synctactic_chunk_word_featurizer, len(featurizer.syntactic_chunk_word)),
        ]

    W_SYNTACTIC_CHUNK_TAG_FEATURIZER=False 
    if W_SYNTACTIC_CHUNK_TAG_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_syntactic_chunk_tag_featurizer, len(featurizer.syntactic_chunk_tag)),
        ]

    W_WORD_TAG_FEATURIZER=False
    if W_WORD_TAG_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_word_tag_featurizer, len(featurizer.word_tag)),
        ]

    W_1_GRAM_TAG_FEATURIZER=False
    if W_1_GRAM_TAG_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_1_gram_tag_featurizer, len(featurizer.tag_ngrams[0])),
        ]

    W_1_GRAM_SYNCTACTIC_CHUNK_FEATURIZER=False
    if W_1_GRAM_SYNCTACTIC_CHUNK_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_1_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[0])),
        ]

    WORD_PROPERTIES=False
    if WORD_PROPERTIES:
        featurizer.word_featurizers+=[
            (featurizer.w_is_number_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
            (featurizer.w_is_capitalized_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
            (featurizer.w_is_split_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
        ]



    ONEGRAM = False
    if ONEGRAM:
        featurizer.word_featurizers+=[
            (featurizer.w_1_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[0])),
            (featurizer.w_1_gram_tag_featurizer, len(featurizer.tag_ngrams[0])),
            (featurizer.w_word_tag_featurizer, len(featurizer.word_tag)),
            (featurizer.w_syntactic_chunk_tag_featurizer, len(featurizer.syntactic_chunk_tag)),
            (featurizer.w_synctactic_chunk_word_featurizer, len(featurizer.syntactic_chunk_word)),
        ]



    W_2_GRAM_TAG_FEATURIZER=False
    if W_2_GRAM_TAG_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_2_gram_tag_featurizer, len(featurizer.tag_ngrams[1])),
        ]

    W_2_GRAM_FORWARD_TAG_FEATURIZER=False
    if W_2_GRAM_FORWARD_TAG_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_2_gram_forward_tag_featurizer, len(featurizer.tag_ngrams[1])),
        ]

    W_2_GRAM_SYNCTACTIC_CHUNK_FEATURIZER=False
    if W_2_GRAM_SYNCTACTIC_CHUNK_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_2_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[1])),
        ]

    W_2_GRAM_FORWARD_SYNCTACTIC_CHUNK_FEATURIZER=False
    if W_2_GRAM_FORWARD_SYNCTACTIC_CHUNK_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_2_gram_forward_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[1])),
        ]

    W_3_GRAM_TAG_FEATURIZER=False
    if W_3_GRAM_TAG_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_3_gram_tag_featurizer, len(featurizer.tag_ngrams[2])),
        ]

    W_3_GRAM_FORWARD_TAG_FEATURIZER=False
    if W_3_GRAM_FORWARD_TAG_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_3_gram_forward_tag_featurizer, len(featurizer.tag_ngrams[2])),
        ]

    W_3_GRAM_AROUND_TAG_FEATURIZER=False
    if W_3_GRAM_AROUND_TAG_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_3_gram_around_tag_featurizer, len(featurizer.tag_ngrams[2])),
        ]

    W_3_GRAM_SYNCTACTIC_CHUNK_FEATURIZER=False
    if W_3_GRAM_SYNCTACTIC_CHUNK_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_3_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
        ]

    W_3_GRAM_FORWARD_SYNCTACTIC_CHUNK_FEATURIZER=False
    if W_3_GRAM_FORWARD_SYNCTACTIC_CHUNK_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_3_gram_forward_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
        ]

    W_3_GRAM_AROUND_SYNCTACTIC_CHUNK_FEATURIZER=False
    if W_3_GRAM_AROUND_SYNCTACTIC_CHUNK_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_3_gram_around_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
        ]

    W_IS_NUMBER_FEATURIZER=False
    if W_IS_NUMBER_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_is_number_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
        ]

    W_IS_CAPITALIZED_FEATURIZER=False
    if W_IS_CAPITALIZED_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_is_capitalized_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
        ]

    W_IS_SPLIT_FEATURIZER=False
    if W_IS_SPLIT_FEATURIZER:
        featurizer.word_featurizers+=[
            (featurizer.w_is_split_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
        ]

    BIGRAM = False
    if BIGRAM:
        featurizer.word_featurizers+=[
            (featurizer.w_2_gram_tag_featurizer, len(featurizer.tag_ngrams[1])),
            (featurizer.w_2_gram_forward_tag_featurizer, len(featurizer.tag_ngrams[1])),
            (featurizer.w_2_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[1])),
            (featurizer.w_2_gram_forward_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[1])),
        ]

    TRIGRAM = False
    if TRIGRAM:
        featurizer.word_featurizers+=[
            (featurizer.w_3_gram_tag_featurizer, len(featurizer.tag_ngrams[2])),
            (featurizer.w_3_gram_forward_tag_featurizer, len(featurizer.tag_ngrams[2])),
            (featurizer.w_3_gram_around_tag_featurizer, len(featurizer.tag_ngrams[2])),
            (featurizer.w_3_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
            (featurizer.w_3_gram_forward_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
            (featurizer.w_3_gram_around_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
        ]

    ALL_FEATURES = True
    if ALL_FEATURES:
        featurizer.word_featurizers=[
            (featurizer.w_is_number_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
            (featurizer.w_is_capitalized_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
            (featurizer.w_is_split_featurizer, len(featurizer.named_entity_recognition_tags.keys())),
            (featurizer.w_1_gram_tag_featurizer, len(featurizer.tag_ngrams[0])),
            (featurizer.w_2_gram_tag_featurizer, len(featurizer.tag_ngrams[1])),
            (featurizer.w_3_gram_tag_featurizer, len(featurizer.tag_ngrams[2])),
            (featurizer.w_2_gram_forward_tag_featurizer, len(featurizer.tag_ngrams[1])),
            (featurizer.w_3_gram_forward_tag_featurizer, len(featurizer.tag_ngrams[2])),
            (featurizer.w_3_gram_around_tag_featurizer, len(featurizer.tag_ngrams[2])),
            (featurizer.w_1_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[0])),
            (featurizer.w_2_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[1])),
            (featurizer.w_3_gram_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
            (featurizer.w_2_gram_forward_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[1])),
            (featurizer.w_3_gram_forward_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
            (featurizer.w_3_gram_around_synctactic_chunk_featurizer, len(featurizer.syntactic_chunk_ngrams[2])),
            (featurizer.w_synctactic_chunk_word_featurizer, len(featurizer.syntactic_chunk_word)),
            (featurizer.w_syntactic_chunk_tag_featurizer, len(featurizer.syntactic_chunk_tag)),
            (featurizer.w_word_tag_featurizer, len(featurizer.word_tag)),
        ]

    EVAL_TRAIN=False
    EVAL_TRAIN_SMALL=True
    FULL_DATASET=False
    SMALL_DATASET=True

    if FULL_DATASET:

        xTrain, featureCountsPerWordTrain = featurizer.featurizeTrain(trainData)
        yTrain = [t[:,3] for t in trainData]
        if EVAL_TRAIN:
            xTrainEval, featureCountsPerWordTrain = featurizer.featurize(trainData)
        xDev, featureCountsPerWordDev = featurizer.featurize(devData)
        xTest, featureCountsPerWordTest = featurizer.featurize(testData)
        perceptron = StructuredPerceptron(featureCountsPerWordTrain, featurizer)

    if SMALL_DATASET:

        xTrainSmall, featureCountsPerWordTrain = featurizer.featurizeTrain(trainDataSmall)
        yTrainSmall = [t[:,3] for t in trainDataSmall]
        if EVAL_TRAIN_SMALL:
            xTrainEvalSmall, featureCountsPerWordTrain = featurizer.featurize(trainDataSmall)
        xDevSmall, featureCountsPerWordDev = featurizer.featurize(devDataSmall)
        xTestSmall, featureCountsPerWordTest = featurizer.featurize(testDataSmall)
        perceptronSmall = StructuredPerceptron(featureCountsPerWordTrain, featurizer)


    for t in range(1, 51):
        logger.info("round %d", t)
        if FULL_DATASET:

            perceptron.train(xTrain, yTrain, T=t)


            if t%10 == 0 or t == 1:
                if EVAL_TRAIN:
                    yTrainPredicted = perceptron.predict(xTrainEval)

                    trainDataOut = zipResults(trainData, yTrainPredicted)
                    print("Round {}, Train set evaluation:".format(t))
                    outResults(trainOutDataPath, trainDataOut)

                yDev = perceptron.predict(xDev)

                devDataOut = zipResults(devData, yDev)
                print("Round {}, Dev set evaluation:".format(t))
                outResults(devOutDataPath, devDataOut)

            if t == 50:
                yTest = perceptron.predict(xTest)

                testDataOut = zipResults(testData, yTest)
                print("Round {}, Test set evaluation:".format(t))
                outResults(testOutDataPath, testDataOut)


        if SMALL_DATASET:

            perceptronSmall.train(xTrainSmall, yTrainSmall, T=t)

            if t%10 == 0 or t == 1:
                if EVAL_TRAIN_SMALL:
                    yTrainSmallPredicted = perceptronSmall.predict(xTrainEvalSmall)
                    trainDataSmallOut = zipResults(trainDataSmall, yTrainSmallPredicted)
                    print("Round {}, Train Small set evaluation:".format(t))
                    outResults(trainSmallOutDataPath, trainDataSmallOut)

                yDevSmall = perceptronSmall.predict(xDevSmall)

                devDataSmallOut = zipResults(devDataSmall, yDevSmall)
                print("Round {}, Dev Small set evaluation:".format(t))
                outResults(devSmallOutDataPath, devDataSmallOut)  

            if t == 50:
                yTestSmall = perceptronSmall.predict(xTestSmall)

                testSmallDataOut = zipResults(testDataSmall, yTestSmall)
                print("Round {}, Test Small set evaluation:".format(t))
                outResults(testSmallOutDataPath, testSmallDataOut)



########## Results #######################

# ALL_FEATURES
# Round 1, Train set evaluation:
# processed 204567 tokens with 23499 phrases; found: 28684 phrases; correct: 14734.
# accuracy:  91.78%; precision:  51.37%; recall:  62.70%; FB1:  56.47
#               LOC: precision:  65.02%; recall:  77.30%; FB1:  70.63  8488
#              MISC: precision:  63.40%; recall:  48.52%; FB1:  54.97  2631
#               ORG: precision:  44.86%; recall:  28.94%; FB1:  35.18  4077
#               PER: precision:  42.39%; recall:  86.64%; FB1:  56.93  13488
# Round 1, Dev set evaluation:
# processed 51578 tokens with 5942 phrases; found: 7162 phrases; correct: 3603.
# accuracy:  91.57%; precision:  50.31%; recall:  60.64%; FB1:  54.99
#               LOC: precision:  60.59%; recall:  69.30%; FB1:  64.65  2101
#              MISC: precision:  61.69%; recall:  48.37%; FB1:  54.22  723
#               ORG: precision:  40.21%; recall:  26.32%; FB1:  31.82  878
#               PER: precision:  44.25%; recall:  83.12%; FB1:  57.75  3460
# Round 1, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5639 phrases; correct: 2147.
# accuracy:  88.67%; precision:  38.07%; recall:  45.53%; FB1:  41.47
#               LOC: precision:  79.36%; recall:  36.73%; FB1:  50.22  654
#              MISC: precision:  48.23%; recall:  44.48%; FB1:  46.28  651
#               ORG: precision:  25.27%; recall:   9.19%; FB1:  13.47  455
#               PER: precision:  30.91%; recall:  89.14%; FB1:  45.90  3879

# Round 1, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 960 phrases; correct: 331.
# accuracy:  88.22%; precision:  34.48%; recall:  40.51%; FB1:  37.25
#               LOC: precision:  70.00%; recall:  27.45%; FB1:  39.44  100
#              MISC: precision:  37.86%; recall:  35.14%; FB1:  36.45  103
#               ORG: precision:  14.29%; recall:   4.15%; FB1:   6.43  56
#               PER: precision:  30.53%; recall:  82.95%; FB1:  44.63  701

# Round 10, Train set evaluation:
# processed 204567 tokens with 23499 phrases; found: 26511 phrases; correct: 18464.
# accuracy:  95.95%; precision:  69.65%; recall:  78.57%; FB1:  73.84
#               LOC: precision:  86.40%; recall:  83.25%; FB1:  84.79  6880
#              MISC: precision:  72.38%; recall:  81.65%; FB1:  76.74  3878
#               ORG: precision:  60.13%; recall:  52.76%; FB1:  56.21  5546
#               PER: precision:  62.49%; recall:  96.64%; FB1:  75.90  10207
# Round 10, Dev set evaluation:
# processed 51578 tokens with 5942 phrases; found: 6690 phrases; correct: 4101.
# accuracy:  94.13%; precision:  61.30%; recall:  69.02%; FB1:  64.93
#               LOC: precision:  79.97%; recall:  68.05%; FB1:  73.53  1563
#              MISC: precision:  61.09%; recall:  66.59%; FB1:  63.73  1005
#               ORG: precision:  49.74%; recall:  42.28%; FB1:  45.71  1140
#               PER: precision:  56.00%; recall:  90.66%; FB1:  69.24  2982

# Round 10, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 4795 phrases; correct: 3469.
# accuracy:  96.19%; precision:  72.35%; recall:  73.56%; FB1:  72.95
#               LOC: precision:  87.22%; recall:  77.28%; FB1:  81.95  1252
#              MISC: precision:  82.11%; recall:  76.06%; FB1:  78.97  654
#               ORG: precision:  59.86%; recall:  41.69%; FB1:  49.15  872
#               PER: precision:  65.34%; recall:  97.99%; FB1:  78.41  2017
# Round 10, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 819 phrases; correct: 469.
# accuracy:  93.16%; precision:  57.26%; recall:  57.41%; FB1:  57.33
#               LOC: precision:  78.12%; recall:  58.82%; FB1:  67.11  192
#              MISC: precision:  69.05%; recall:  52.25%; FB1:  59.49  84
#               ORG: precision:  34.40%; recall:  22.28%; FB1:  27.04  125
#               PER: precision:  52.15%; recall:  84.50%; FB1:  64.50  418


# Round 20, Train set evaluation:
# processed 204567 tokens with 23499 phrases; found: 24681 phrases; correct: 18505.
# accuracy:  96.71%; precision:  74.98%; recall:  78.75%; FB1:  76.82
#               LOC: precision:  85.56%; recall:  80.41%; FB1:  82.90  6710
#              MISC: precision:  77.13%; recall:  83.60%; FB1:  80.23  3726
#               ORG: precision:  65.58%; recall:  53.85%; FB1:  59.14  5191
#               PER: precision:  71.64%; recall:  98.27%; FB1:  82.87  9054
# Round 20, Dev set evaluation:
# processed 51578 tokens with 5942 phrases; found: 6130 phrases; correct: 4012.
# accuracy:  94.57%; precision:  65.45%; recall:  67.52%; FB1:  66.47
#               LOC: precision:  81.00%; recall:  65.00%; FB1:  72.12  1474
#              MISC: precision:  64.30%; recall:  67.79%; FB1:  66.00  972
#               ORG: precision:  53.27%; recall:  39.52%; FB1:  45.38  995
#               PER: precision:  61.84%; recall:  90.28%; FB1:  73.41  2689
# Round 20, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 4700 phrases; correct: 3798.
# accuracy:  97.51%; precision:  80.81%; recall:  80.53%; FB1:  80.67
#               LOC: precision:  81.46%; recall:  97.31%; FB1:  88.68  1688
#              MISC: precision:  81.22%; recall:  64.31%; FB1:  71.78  559
#               ORG: precision:  74.94%; recall:  74.28%; FB1:  74.61  1241
#               PER: precision:  85.73%; recall:  77.25%; FB1:  81.27  1212
# Round 20, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 819 phrases; correct: 435.
# accuracy:  92.84%; precision:  53.11%; recall:  53.24%; FB1:  53.18
#               LOC: precision:  56.90%; recall:  79.22%; FB1:  66.23  355
#              MISC: precision:  62.67%; recall:  42.34%; FB1:  50.54  75
#               ORG: precision:  37.38%; recall:  41.45%; FB1:  39.31  214
#               PER: precision:  60.57%; recall:  41.09%; FB1:  48.96  175

# Round 30, Train set evaluation:
# processed 204567 tokens with 23499 phrases; found: 24471 phrases; correct: 19325.
# accuracy:  97.25%; precision:  78.97%; recall:  82.24%; FB1:  80.57
#               LOC: precision:  83.32%; recall:  93.80%; FB1:  88.25  8038
#              MISC: precision:  81.55%; recall:  83.07%; FB1:  82.31  3502
#               ORG: precision:  66.42%; recall:  52.93%; FB1:  58.91  5038
#               PER: precision:  81.41%; recall:  97.36%; FB1:  88.68  7893
# Round 30, Dev set evaluation:
# processed 51578 tokens with 5942 phrases; found: 6155 phrases; correct: 4202.
# accuracy:  95.02%; precision:  68.27%; recall:  70.72%; FB1:  69.47
#               LOC: precision:  74.65%; recall:  80.13%; FB1:  77.29  1972
#              MISC: precision:  70.83%; recall:  66.38%; FB1:  68.53  864
#               ORG: precision:  51.86%; recall:  39.52%; FB1:  44.86  1022
#               PER: precision:  69.13%; recall:  86.21%; FB1:  76.73  2297
# Round 30, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 4511 phrases; correct: 3893.
# accuracy:  97.80%; precision:  86.30%; recall:  82.55%; FB1:  84.38
#               LOC: precision:  90.25%; recall:  94.34%; FB1:  92.25  1477
#              MISC: precision:  86.67%; recall:  75.50%; FB1:  80.70  615
#               ORG: precision:  74.40%; recall:  66.85%; FB1:  70.42  1125
#               PER: precision:  91.96%; recall:  88.48%; FB1:  90.19  1294
# Round 30, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 744 phrases; correct: 434.
# accuracy:  93.00%; precision:  58.33%; recall:  53.12%; FB1:  55.61
#               LOC: precision:  61.74%; recall:  72.16%; FB1:  66.55  298
#              MISC: precision:  64.00%; recall:  43.24%; FB1:  51.61  75
#               ORG: precision:  38.75%; recall:  32.12%; FB1:  35.13  160
#               PER: precision:  66.35%; recall:  54.26%; FB1:  59.70  211

# Round 40, Train set evaluation:
# processed 204567 tokens with 23499 phrases; found: 25049 phrases; correct: 19439.
# accuracy:  97.28%; precision:  77.60%; recall:  82.72%; FB1:  80.08
#               LOC: precision:  87.27%; recall:  84.59%; FB1:  85.91  6921
#              MISC: precision:  75.97%; recall:  90.29%; FB1:  82.51  4086
#               ORG: precision:  68.25%; recall:  59.83%; FB1:  63.77  5541
#               PER: precision:  76.61%; recall:  98.68%; FB1:  86.26  8501
# Round 40, Dev set evaluation:
# processed 51578 tokens with 5942 phrases; found: 6402 phrases; correct: 4157.
# accuracy:  94.80%; precision:  64.93%; recall:  69.96%; FB1:  67.35
#               LOC: precision:  80.66%; recall:  68.10%; FB1:  73.85  1551
#              MISC: precision:  55.72%; recall:  71.37%; FB1:  62.58  1181
#               ORG: precision:  52.31%; recall:  44.82%; FB1:  48.27  1149
#               PER: precision:  65.33%; recall:  89.41%; FB1:  75.50  2521

# Round 40, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 4811 phrases; correct: 4326.
# accuracy:  98.95%; precision:  89.92%; recall:  91.73%; FB1:  90.82
#               LOC: precision:  89.68%; recall:  97.74%; FB1:  93.53  1540
#              MISC: precision:  89.33%; recall:  88.95%; FB1:  89.14  703
#               ORG: precision:  83.46%; recall:  81.39%; FB1:  82.41  1221
#               PER: precision:  96.36%; recall:  96.51%; FB1:  96.43  1347

# Round 40, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 872 phrases; correct: 504.
# accuracy:  93.78%; precision:  57.80%; recall:  61.69%; FB1:  59.68
#               LOC: precision:  60.25%; recall:  76.08%; FB1:  67.24  322
#              MISC: precision:  54.81%; recall:  51.35%; FB1:  53.02  104
#               ORG: precision:  41.43%; recall:  45.08%; FB1:  43.18  210
#               PER: precision:  70.34%; recall:  64.34%; FB1:  67.21  236

# Round 50, Train set evaluation:
# processed 204567 tokens with 23499 phrases; found: 25672 phrases; correct: 19153.
# accuracy:  96.91%; precision:  74.61%; recall:  81.51%; FB1:  77.90
#               LOC: precision:  86.87%; recall:  87.41%; FB1:  87.14  7184
#              MISC: precision:  56.56%; recall:  94.68%; FB1:  70.81  5755
#               ORG: precision:  66.49%; recall:  50.48%; FB1:  57.39  4799
#               PER: precision:  81.50%; recall:  97.97%; FB1:  88.98  7934
# Round 50, Dev set evaluation:
# processed 51578 tokens with 5942 phrases; found: 6706 phrases; correct: 4112.
# accuracy:  94.28%; precision:  61.32%; recall:  69.20%; FB1:  65.02
#               LOC: precision:  78.22%; recall:  70.39%; FB1:  74.10  1653
#              MISC: precision:  40.21%; recall:  77.55%; FB1:  52.96  1778
#               ORG: precision:  54.92%; recall:  37.88%; FB1:  44.84  925
#               PER: precision:  67.91%; recall:  86.64%; FB1:  76.15  2350

# Round 50, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 4898 phrases; correct: 4216.
# accuracy:  98.67%; precision:  86.08%; recall:  89.40%; FB1:  87.71
#               LOC: precision:  88.08%; recall:  96.74%; FB1:  92.21  1552
#              MISC: precision:  90.37%; recall:  90.37%; FB1:  90.37  706
#               ORG: precision:  79.45%; recall:  88.34%; FB1:  83.66  1392
#               PER: precision:  88.54%; recall:  82.16%; FB1:  85.23  1248
# Round 50, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 899 phrases; correct: 451.
# accuracy:  92.58%; precision:  50.17%; recall:  55.20%; FB1:  52.56
#               LOC: precision:  56.16%; recall:  73.33%; FB1:  63.61  333
#              MISC: precision:  62.22%; recall:  50.45%; FB1:  55.72  90
#               ORG: precision:  34.19%; recall:  55.44%; FB1:  42.29  313
#               PER: precision:  61.96%; recall:  39.15%; FB1:  47.98  163

# Round 50, Test set evaluation:
# processed 46666 tokens with 5648 phrases; found: 6373 phrases; correct: 3474.
# accuracy:  92.47%; precision:  54.51%; recall:  61.51%; FB1:  57.80
#               LOC: precision:  76.80%; recall:  69.84%; FB1:  73.16  1517
#              MISC: precision:  30.30%; recall:  71.51%; FB1:  42.56  1657
#               ORG: precision:  53.50%; recall:  28.96%; FB1:  37.58  899
#               PER: precision:  57.65%; recall:  82.00%; FB1:  67.70  2300

# Round 50, Test Small set evaluation:
# processed 6288 tokens with 796 phrases; found: 921 phrases; correct: 416.
# accuracy:  90.30%; precision:  45.17%; recall:  52.26%; FB1:  48.46
#               LOC: precision:  54.79%; recall:  80.65%; FB1:  65.25  365
#              MISC: precision:  38.68%; recall:  37.61%; FB1:  38.14  106
#               ORG: precision:  30.67%; recall:  42.02%; FB1:  35.46  326
#               PER: precision:  60.48%; recall:  37.31%; FB1:  46.15  124



# Abalation Study

# BIGRAM, ONEGRAM AND WORD_PROPERTIES

# Round 1, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5216 phrases; correct: 2005.
# accuracy:  89.67%; precision:  38.44%; recall:  42.51%; FB1:  40.37
#               LOC: precision:  73.55%; recall:  39.56%; FB1:  51.45  760
#              MISC: precision:  49.00%; recall:  27.76%; FB1:  35.44  400
#               ORG: precision:  18.29%; recall:   8.55%; FB1:  11.65  585
#               PER: precision:  32.93%; recall:  84.98%; FB1:  47.47  3471
# Round 1, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 900 phrases; correct: 334.
# accuracy:  89.74%; precision:  37.11%; recall:  40.88%; FB1:  38.91
#               LOC: precision:  67.18%; recall:  34.51%; FB1:  45.60  131
#              MISC: precision:  40.00%; recall:  21.62%; FB1:  28.07  60
#               ORG: precision:  11.11%; recall:   4.15%; FB1:   6.04  72
#               PER: precision:  33.59%; recall:  82.95%; FB1:  47.82  637

# Round 10, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5306 phrases; correct: 3277.
# accuracy:  94.71%; precision:  61.76%; recall:  69.49%; FB1:  65.40
#               LOC: precision:  79.97%; recall:  77.99%; FB1:  78.97  1378
#              MISC: precision:  78.35%; recall:  79.46%; FB1:  78.90  716
#               ORG: precision:  53.65%; recall:  25.80%; FB1:  34.84  602
#               PER: precision:  49.46%; recall:  95.99%; FB1:  65.28  2610
# Round 10, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 924 phrases; correct: 484.
# accuracy:  92.36%; precision:  52.38%; recall:  59.24%; FB1:  55.60
#               LOC: precision:  76.53%; recall:  63.92%; FB1:  69.66  213
#              MISC: precision:  77.38%; recall:  58.56%; FB1:  66.67  84
#               ORG: precision:  40.68%; recall:  12.44%; FB1:  19.05  59
#               PER: precision:  40.85%; recall:  89.92%; FB1:  56.17  568

# Round 20, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5256 phrases; correct: 3862.
# accuracy:  97.04%; precision:  73.48%; recall:  81.89%; FB1:  77.46
#               LOC: precision:  79.12%; recall:  91.72%; FB1:  84.96  1638
#              MISC: precision:  84.01%; recall:  81.87%; FB1:  82.93  688
#               ORG: precision:  65.48%; recall:  52.88%; FB1:  58.51  1011
#               PER: precision:  69.10%; recall:  98.59%; FB1:  81.25  1919
# Round 20, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 954 phrases; correct: 525.
# accuracy:  93.46%; precision:  55.03%; recall:  64.26%; FB1:  59.29
#               LOC: precision:  62.46%; recall:  75.69%; FB1:  68.44  309
#              MISC: precision:  77.78%; recall:  56.76%; FB1:  65.62  81
#               ORG: precision:  30.23%; recall:  20.21%; FB1:  24.22  129
#               PER: precision:  52.87%; recall:  89.15%; FB1:  66.38  435
# Round 30, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5088 phrases; correct: 3739.
# accuracy:  96.60%; precision:  73.49%; recall:  79.28%; FB1:  76.27
#               LOC: precision:  81.06%; recall:  89.03%; FB1:  84.86  1552
#              MISC: precision:  75.51%; recall:  93.48%; FB1:  83.54  874
#               ORG: precision:  62.50%; recall:  38.74%; FB1:  47.83  776
#               PER: precision:  70.84%; recall:  99.33%; FB1:  82.70  1886
# Round 30, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 931 phrases; correct: 509.
# accuracy:  92.93%; precision:  54.67%; recall:  62.30%; FB1:  58.24
#               LOC: precision:  74.10%; recall:  72.94%; FB1:  73.52  251
#              MISC: precision:  55.91%; recall:  63.96%; FB1:  59.66  127
#               ORG: precision:  43.06%; recall:  16.06%; FB1:  23.40  72
#               PER: precision:  45.95%; recall:  85.66%; FB1:  59.81  481

# Round 40, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5190 phrases; correct: 4002.
# accuracy:  97.60%; precision:  77.11%; recall:  84.86%; FB1:  80.80
#               LOC: precision:  75.66%; recall:  96.82%; FB1:  84.94  1808
#              MISC: precision:  83.92%; recall:  87.25%; FB1:  85.56  734
#               ORG: precision:  68.30%; recall:  54.71%; FB1:  60.75  1003
#               PER: precision:  81.03%; recall:  99.11%; FB1:  89.16  1645
# Round 40, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 986 phrases; correct: 538.
# accuracy:  93.27%; precision:  54.56%; recall:  65.85%; FB1:  59.68
#               LOC: precision:  60.57%; recall:  83.14%; FB1:  70.08  350
#              MISC: precision:  71.74%; recall:  59.46%; FB1:  65.02  92
#               ORG: precision:  34.19%; recall:  20.73%; FB1:  25.81  117
#               PER: precision:  51.52%; recall:  85.27%; FB1:  64.23  427

# Round 50, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 4989 phrases; correct: 3925.
# accuracy:  97.80%; precision:  78.67%; recall:  83.23%; FB1:  80.89
#               LOC: precision:  85.01%; recall:  85.07%; FB1:  85.04  1414
#              MISC: precision:  78.96%; recall:  92.49%; FB1:  85.19  827
#               ORG: precision:  66.00%; recall:  76.92%; FB1:  71.04  1459
#               PER: precision:  85.88%; recall:  82.30%; FB1:  84.05  1289
# Round 50, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 937 phrases; correct: 417.
# accuracy:  91.80%; precision:  44.50%; recall:  51.04%; FB1:  47.55
#               LOC: precision:  74.48%; recall:  69.80%; FB1:  72.06  239
#              MISC: precision:  33.18%; recall:  66.67%; FB1:  44.31  223
#               ORG: precision:  28.29%; recall:  37.82%; FB1:  32.37  258
#               PER: precision:  42.40%; recall:  35.66%; FB1:  38.74  217
# Round 50, Test Small set evaluation:
# processed 6288 tokens with 796 phrases; found: 954 phrases; correct: 337.
# accuracy:  89.01%; precision:  35.32%; recall:  42.34%; FB1:  38.51
#               LOC: precision:  70.42%; recall:  68.15%; FB1:  69.26  240
#              MISC: precision:  21.72%; recall:  53.21%; FB1:  30.85  267
#               ORG: precision:  25.17%; recall:  31.51%; FB1:  27.99  298
#               PER: precision:  23.49%; recall:  17.41%; FB1:  20.00  149

# ONEGRAM AND WORD_PROPERTIES

# Round 1, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5074 phrases; correct: 2418.
# accuracy:  90.75%; precision:  47.65%; recall:  51.27%; FB1:  49.40
#               LOC: precision:  67.86%; recall:  70.98%; FB1:  69.39  1478
#              MISC: precision:  92.86%; recall:  31.30%; FB1:  46.82  238
#               ORG: precision:  21.74%; recall:   2.40%; FB1:   4.32  138
#               PER: precision:  36.15%; recall:  86.54%; FB1:  51.00  3220
# Round 1, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 860 phrases; correct: 398.
# accuracy:  90.77%; precision:  46.28%; recall:  48.71%; FB1:  47.47
#               LOC: precision:  66.82%; recall:  56.86%; FB1:  61.44  217
#              MISC: precision:  90.00%; recall:  24.32%; FB1:  38.30  30
#               ORG: precision:  35.00%; recall:   3.63%; FB1:   6.57  20
#               PER: precision:  36.93%; recall:  84.88%; FB1:  51.47  593

# Round 10, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5073 phrases; correct: 3399.
# accuracy:  95.80%; precision:  67.00%; recall:  72.07%; FB1:  69.45
#               LOC: precision:  83.53%; recall:  59.94%; FB1:  69.80  1014
#              MISC: precision:  80.39%; recall:  81.30%; FB1:  80.85  714
#               ORG: precision:  63.72%; recall:  51.20%; FB1:  56.78  1006
#               PER: precision:  57.16%; recall:  99.41%; FB1:  72.58  2339
# Round 10, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 925 phrases; correct: 471.
# accuracy:  92.55%; precision:  50.92%; recall:  57.65%; FB1:  54.08
#               LOC: precision:  80.39%; recall:  48.24%; FB1:  60.29  153
#              MISC: precision:  79.76%; recall:  60.36%; FB1:  68.72  84
#               ORG: precision:  43.12%; recall:  24.35%; FB1:  31.13  109
#               PER: precision:  40.41%; recall:  90.70%; FB1:  55.91  579
# Round 20, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5086 phrases; correct: 3792.
# accuracy:  96.98%; precision:  74.56%; recall:  80.41%; FB1:  77.37
#               LOC: precision:  85.67%; recall:  66.03%; FB1:  74.58  1089
#              MISC: precision:  81.25%; recall:  75.50%; FB1:  78.27  656
#               ORG: precision:  65.20%; recall:  79.47%; FB1:  71.63  1526
#               PER: precision:  73.33%; recall:  98.96%; FB1:  84.24  1815
# Round 20, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 983 phrases; correct: 442.
# accuracy:  91.91%; precision:  44.96%; recall:  54.10%; FB1:  49.11
#               LOC: precision:  81.48%; recall:  51.76%; FB1:  63.31  162
#              MISC: precision:  80.72%; recall:  60.36%; FB1:  69.07  83
#               ORG: precision:  26.70%; recall:  48.70%; FB1:  34.50  352
#               PER: precision:  38.60%; recall:  57.75%; FB1:  46.27  386
# Round 30, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5015 phrases; correct: 3854.
# accuracy:  97.26%; precision:  76.85%; recall:  81.72%; FB1:  79.21
#               LOC: precision:  86.18%; recall:  67.94%; FB1:  75.98  1114
#              MISC: precision:  80.43%; recall:  68.70%; FB1:  74.10  603
#               ORG: precision:  62.49%; recall:  86.90%; FB1:  72.70  1741
#               PER: precision:  84.84%; recall:  98.22%; FB1:  91.04  1557
# Round 30, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 928 phrases; correct: 458.
# accuracy:  92.84%; precision:  49.35%; recall:  56.06%; FB1:  52.49
#               LOC: precision:  83.93%; recall:  55.29%; FB1:  66.67  168
#              MISC: precision:  79.76%; recall:  60.36%; FB1:  68.72  84
#               ORG: precision:  27.91%; recall:  53.37%; FB1:  36.65  369
#               PER: precision:  47.88%; recall:  56.98%; FB1:  52.04  307
# Round 40, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5095 phrases; correct: 4045.
# accuracy:  97.66%; precision:  79.39%; recall:  85.77%; FB1:  82.46
#               LOC: precision:  78.68%; recall:  93.49%; FB1:  85.45  1679
#              MISC: precision:  81.57%; recall:  79.60%; FB1:  80.57  689
#               ORG: precision:  66.05%; recall:  71.17%; FB1:  68.51  1349
#               PER: precision:  92.24%; recall:  94.50%; FB1:  93.35  1378
# Round 40, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 922 phrases; correct: 444.
# accuracy:  92.22%; precision:  48.16%; recall:  54.35%; FB1:  51.06
#               LOC: precision:  73.75%; recall:  69.41%; FB1:  71.52  240
#              MISC: precision:  81.25%; recall:  58.56%; FB1:  68.06  80
#               ORG: precision:  24.83%; recall:  56.48%; FB1:  34.49  439
#               PER: precision:  57.06%; recall:  36.05%; FB1:  44.18  163
# Round 50, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 5196 phrases; correct: 3757.
# accuracy:  96.58%; precision:  72.31%; recall:  79.66%; FB1:  75.81
#               LOC: precision:  79.10%; recall:  92.14%; FB1:  85.13  1646
#              MISC: precision:  82.53%; recall:  74.93%; FB1:  78.54  641
#               ORG: precision:  68.21%; recall:  48.16%; FB1:  56.46  884
#               PER: precision:  65.33%; recall:  98.36%; FB1:  78.52  2025
# Round 50, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 876 phrases; correct: 519.
# accuracy:  93.67%; precision:  59.25%; recall:  63.53%; FB1:  61.31
#               LOC: precision:  76.07%; recall:  69.80%; FB1:  72.80  234
#              MISC: precision:  81.71%; recall:  60.36%; FB1:  69.43  82
#               ORG: precision:  48.81%; recall:  21.24%; FB1:  29.60  84
#               PER: precision:  48.95%; recall:  90.31%; FB1:  63.49  476
# Round 50, Test Small set evaluation:
# processed 6288 tokens with 796 phrases; found: 913 phrases; correct: 417.
# accuracy:  90.47%; precision:  45.67%; recall:  52.39%; FB1:  48.80
#               LOC: precision:  69.48%; recall:  69.76%; FB1:  69.62  249
#              MISC: precision:  60.56%; recall:  39.45%; FB1:  47.78  71
#               ORG: precision:  37.89%; recall:  15.13%; FB1:  21.62  95
#               PER: precision:  33.13%; recall:  82.09%; FB1:  47.21  498

# WORD_PROPERTIES
# Round 1, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.48%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0

# Round 1, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.91%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0

# Round 10, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.48%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0

# Round 10, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.91%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0

# Round 20, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.48%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
# Round 20, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.91%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0

# Round 30, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.48%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
# Round 30, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.91%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0

# Round 40, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.48%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0

# Round 40, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.91%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0

# Round 50, Train Small set evaluation:
# processed 41723 tokens with 4716 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.48%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
# Round 50, Dev Small set evaluation:
# processed 7342 tokens with 817 phrases; found: 0 phrases; correct: 0.
# accuracy:  83.91%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
# Round 50, Test Small set evaluation:
# processed 6288 tokens with 796 phrases; found: 0 phrases; correct: 0.
# accuracy:  82.06%; precision:   0.00%; recall:   0.00%; FB1:   0.00
#               LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#              MISC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               ORG: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
#               PER: precision:   0.00%; recall:   0.00%; FB1:   0.00  0