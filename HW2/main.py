import Utilities.Dataset.HW2.TwitterDataset as TwitterDataset
import Utilities.LanguageModel.HMM as HMM
import itertools
from datetime import datetime
import os
import logging
import sys
import numpy as np
import joblib 
from tqdm.auto import tqdm

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
np.set_printoptions(threshold=sys.maxsize)

format = "%(asctime)s: %(message)s"
logfileName = os.path.join("logs", "HW2", datetime.now().strftime('log_%H_%M_%d_%m_%Y.log') )
logging.basicConfig(filename=logfileName, filemode='a', format=format, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger('__HW2__')


class ProgressParallel(joblib.Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total, desc="Outer") as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def trainAndPredict(param, train, progressbar):
    hmm_param = param[0]
    hmm_param.train(train)
    accuracy = hmm_param.predict(eval_corpus=param[1], n=param[2], K=param[3], lambdas=param[4], useLinearInterpolation=param[5],progressbar=progressbar + 1)
    logger.info("i: %d, Params: %s, accuracy: %0.20f", progressbar, param[2:], accuracy)
    return (params[1:], accuracy)


if __name__ == "__main__":


    trainData = TwitterDataset.loadRawTrainData()
    devData = TwitterDataset.loadRawDevData()
    testData = TwitterDataset.loadRawTestData()

    DEVEL = False
    if DEVEL:

        hmm = HMM.HMM(max_n=3)
        hmm.train(trainData)

        hmm.predict(devData, n=3, K=0.0001, lambdas=[0.1, 0.4, 0.5], useLinearInterpolation=True)
        # accuracy: 0.8341928724026725
        hmm.predict(devData, n=3, K=0.001, lambdas=[0.1, 0.4, 0.5], useLinearInterpolation=True)
        # accuracy: 0.8271850208874798
        hmm.predict(devData, n=3, K=0.01, lambdas=[0.1, 0.4, 0.5], useLinearInterpolation=True)
        # accuracy: 0.8042292043707221
        hmm.predict(devData, n=3, K=0.1, lambdas=[0.1, 0.4, 0.5], useLinearInterpolation=True)
        # accuracy: 0.737457306535672
        hmm.predict(devData, n=3, K=1, lambdas=[0.1, 0.4, 0.5], useLinearInterpolation=True)
        # accuracy: 0.5894079386030563

        hmm = HMM.HMM(max_n=2)
        hmm.train(trainData)

        # hmm.predict(trainData, n=2, K=0.0001, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # # accuracy: 0.9628296392723785
        hmm.predict(trainData, n=2, K=0.001, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # accuracy: 0.9624336268027817
        hmm.predict(trainData, n=2, K=0.01, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # accuracy: 0.9578733171182899
        hmm.predict(trainData, n=2, K=0.1, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # accuracy: 
        hmm.predict(trainData, n=2, K=1, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # # accuracy: 


        hmm.predict(trainData, n=2, K=0.0001, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # accuracy: 0.9628296392723785
        hmm.predict(trainData, n=2, K=0.0001, lambdas=[0.10, 0.90], useLinearInterpolation=True)
        # accuracy: 0.9628296392723785
        hmm.predict(trainData, n=2, K=0.0001, lambdas=[0.15, 0.85], useLinearInterpolation=True)
        # accuracy: 0.9628296392723785
        hmm.predict(trainData, n=2, K=0.0001, lambdas=[0.20, 0.80], useLinearInterpolation=True)
        # accuracy: 0.9628296392723785
        hmm.predict(trainData, n=2, K=0.0001, lambdas=[0.25, 0.75], useLinearInterpolation=True)
        # accuracy: 0.9628296392723785

        hmm.predict(devData, n=2, K=0.0001, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # accuracy: 0.9008559104083604
        hmm.predict(devData, n=2, K=0.001, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # accuracy: 0.9008014804936794
        hmm.predict(devData, n=2, K=0.01, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # accuracy: 0.8996312373280355
        hmm.predict(devData, n=2, K=0.1, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # accuracy: 0.8920246567513506
        hmm.predict(devData, n=2, K=1, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # accuracy: 0.855080352161548

        hmm.predict(devData, n=2, K=0.0001, lambdas=[0.05, 0.95], useLinearInterpolation=True)
        # accuracy: 0.9008559104083604
        hmm.predict(devData, n=2, K=0.0001, lambdas=[0.10, 0.90], useLinearInterpolation=True)
        # accuracy: 0.9008559104083604
        hmm.predict(devData, n=2, K=0.0001, lambdas=[0.15, 0.85], useLinearInterpolation=True)
        # accuracy: 0.9008559104083604
        hmm.predict(devData, n=2, K=0.0001, lambdas=[0.20, 0.80], useLinearInterpolation=True)
        # accuracy: 0.9008559104083604
        hmm.predict(devData, n=2, K=0.0001, lambdas=[0.25, 0.75], useLinearInterpolation=True)

        hmm = HMM.HMM(max_n=3)
        hmm.train(trainData)

        hmm.predict(trainData, n=3, K=0.0001, lambdas=[0.1, 0.4, 0.5])
        hmm.predict(trainData, n=3, K=0.001, lambdas=[0.1, 0.4, 0.5])
        hmm.predict(trainData, n=3, K=0.01, lambdas=[0.1, 0.4, 0.5])
        hmm.predict(trainData, n=3, K=0.1, lambdas=[0.1, 0.4, 0.5])
        hmm.predict(trainData, n=3, K=1, lambdas=[0.1, 0.4, 0.5])


    params = []

    BIGRAM = True
    if BIGRAM:


        TRAIN = False
        if TRAIN:
            params.extend([
                [HMM.HMM(max_n=2), trainData, 2, 0.01, [0.1, 0.9], True],
                # accuracy:  0.96454112568942484973
                [HMM.HMM(max_n=2), trainData, 2, 0.01, [0.2, 0.8], True],
                # accuracy:  0.96527285807269358475
                [HMM.HMM(max_n=2), trainData, 2, 0.01, [0.25, 0.75], True],
                # accuracy: 0.96539344318454323179
                [HMM.HMM(max_n=2), trainData, 2, 0.01, [0.3, 0.7], True],
                # accuracy: 0.96534822376759965579
                [HMM.HMM(max_n=2), trainData, 2, 0.01, [0.35, 0.65], True]
                # accuracy: 0.96483573704223901668
            ])


        DEV = True
        if DEV:
            params.extend([
                [HMM.HMM(max_n=2), devData, 2, 0.01, [0.1, 0.9], True],
                # accuracy: 0.90473404182938943485
                [HMM.HMM(max_n=2), devData, 2, 0.01, [0.2, 0.8], True],
                # accuracy: 0.90412170528922697343
                [HMM.HMM(max_n=2), devData, 2, 0.01, [0.25, 0.75], True],
                # accuracy: 0.90334607900502117417
                [HMM.HMM(max_n=2), devData, 2, 0.01, [0.3, 0.7], True],
                # accuracy: 0.90225748071139899587
                [HMM.HMM(max_n=2), devData, 2, 0.01, [0.35, 0.65], True]
                # accuracy: 0.90160432173522564447
            ])

        TEST = False
        if TEST:
            params.extend([[HMM.HMM(max_n=2), testData, 2, 0.01, [0.1, 0.9], True]])
            # accuracy: 0.90518613223073052243

    TRIGRAM = False

    if TRIGRAM:

        TRAIN = True
        if TRAIN:
            params.extend([
                [HMM.HMM(max_n=3), trainData, 3, 0.01, [0.1, 0.4, 0.5], True],
                # accuracy: 0.95097393032098931354
                [HMM.HMM(max_n=3), trainData, 3, 0.01, [0.1, 0.3, 0.6], True],
                # accuracy: 0.94941180500839295053
                [HMM.HMM(max_n=3), trainData, 3, 0.01, [0.15, 0.35, 0.5], True],
                # accuracy: 0.95382138330307286722
                [HMM.HMM(max_n=3), trainData, 3, 0.01, [0.2, 0.3, 0.5], True],
                # accuracy: 0.94642458291939290937
                [HMM.HMM(max_n=3), trainData, 3, 0.01, [0.15, 0.15, 0.7], True],
                # accuracy: 0.94976533863177003969
            ])

        DEV = False
        if DEV:
            params.extend([
                [HMM.HMM(max_n=3), devData, 3, 0.01, [0.1, 0.4, 0.5], True],
                # accuracy: 0.88524813237355248763
                [HMM.HMM(max_n=3), devData, 3, 0.01, [0.1, 0.3, 0.6], True],
                # accuracy: 0.88481269305610366072
                [HMM.HMM(max_n=3), devData, 3, 0.01, [0.15, 0.35, 0.5], True],
                # accuracy: 0.88554749690429857001
                [HMM.HMM(max_n=3), devData, 3, 0.01, [0.20, 0.30, 0.5], True],
                # accuracy: 0.88601015117908799024
                [HMM.HMM(max_n=3), devData, 3, 0.01, [0.15, 0.15, 0.7], True],
                # accuracy: 0.88601015117908799024
            ])

        TEST = True
        if TEST:
            params.extend([[HMM.HMM(max_n=3), testData, 3, 0.01, [0.20, 0.30, 0.5], True]])
            # accuracy: 0.88694694177332267238

    for i,param in enumerate(params):
        logger.info("%d %d %0.10f %s %s", i, param[2], param[3], param[4], param[5])

    results = ProgressParallel(n_jobs=20, total=len(params))(joblib.delayed(trainAndPredict)(param, trainData, i) for (i, param) in enumerate(params))

    tqdm.write("Completed")
    for result in results:
        logger.info(result)

# ./9187_progress_log_23_34_31_01_2021.log:23:41:15: i: 0, Params: [2, 0.01, [0.1, 0.9], True], accuracy: 0.90473404182938943485
# ./9192_progress_log_23_34_31_01_2021.log:23:41:18: i: 1, Params: [2, 0.01, [0.2, 0.8], True], accuracy: 0.90412170528922697343
# ./9204_progress_log_23_34_31_01_2021.log:23:41:17: i: 2, Params: [2, 0.01, [0.25, 0.75], True], accuracy: 0.90334607900502117417
# ./9191_progress_log_23_34_31_01_2021.log:23:41:20: i: 3, Params: [2, 0.01, [0.3, 0.7], True], accuracy: 0.90225748071139899587
# ./9202_progress_log_23_34_31_01_2021.log:23:41:22: i: 4, Params: [2, 0.01, [0.35, 0.65], True], accuracy: 0.90160432173522564447
# ./9190_progress_log_23_34_31_01_2021.log:00:46:44: i: 5, Params: [2, 0.01, [0.1, 0.9], True], accuracy: 0.96454112568942484973
# ./9197_progress_log_23_34_31_01_2021.log:00:47:01: i: 6, Params: [2, 0.01, [0.2, 0.8], True], accuracy: 0.96527285807269358475
# ./9203_progress_log_23_34_31_01_2021.log:00:47:05: i: 7, Params: [2, 0.01, [0.25, 0.75], True], accuracy: 0.96539344318454323179
# ./9200_progress_log_23_34_31_01_2021.log:00:47:11: i: 8, Params: [2, 0.01, [0.3, 0.7], True], accuracy: 0.96534822376759965579
# ./9198_progress_log_23_34_31_01_2021.log:00:47:21: i: 9, Params: [2, 0.01, [0.35, 0.65], True], accuracy: 0.96483573704223901668
# ./9201_progress_log_23_34_31_01_2021.log:01:34:28: i: 10, Params: [3, 0.01, [0.1, 0.4, 0.5], True], accuracy: 0.88524813237355248763
# ./9189_progress_log_23_34_31_01_2021.log:01:34:21: i: 11, Params: [3, 0.01, [0.1, 0.3, 0.6], True], accuracy: 0.88481269305610366072
# ./9196_progress_log_23_34_31_01_2021.log:01:34:17: i: 12, Params: [3, 0.01, [0.15, 0.35, 0.5], True], accuracy: 0.88554749690429857001
# ./9186_progress_log_23_34_31_01_2021.log:01:33:50: i: 13, Params: [3, 0.01, [0.2, 0.3, 0.5], True], accuracy: 0.88601015117908799024
# ./9195_progress_log_23_34_31_01_2021.log:01:34:19: i: 14, Params: [3, 0.01, [0.25, 0.25, 0.5], True], accuracy: 0.88601015117908799024
# ./9205_progress_log_23_34_31_01_2021.log:12:13:56: i: 15, Params: [3, 0.01, [0.1, 0.4, 0.5], True], accuracy: 0.95097393032098931354
# ./9188_progress_log_23_34_31_01_2021.log:12:07:11: i: 16, Params: [3, 0.01, [0.1, 0.3, 0.6], True], accuracy: 0.94941180500839295053
# ./42133_progress_log_12_22_01_02_2021.log:22:02:11: i: 1, Params: [3, 0.01, [0.2, 0.3, 0.5], True], accuracy: 0.95382138330307286722
# ./9199_progress_log_23_34_31_01_2021.log:12:08:15: i: 18, Params: [3, 0.01, [0.1, 0.2, 0.7], True], accuracy: 0.94642458291939290937
# ./9193_progress_log_23_34_31_01_2021.log:12:09:26: i: 19, Params: [3, 0.01, [0.15, 0.15, 0.7], True], accuracy: 0.94976533863177003969



## Test accuracies:
# ./42122_progress_log_12_22_01_02_2021.log:12:24:45: i: 0, Params: [2, 0.01, [0.1, 0.9], True], accuracy: 0.90518613223073052243
# ./42123_progress_log_12_22_01_02_2021.log:13:37:03: i: 2, Params: [3, 0.01, [0.2, 0.3, 0.5], True], accuracy: 0.88694694177332267238
