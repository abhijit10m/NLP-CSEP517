Homework-3

To run the assignment, execute `python3.8 HW3/main.py`

the file "main.py" is divided into 3 sections :
0. DEVEL (Alpha testing)
1. FULL_DATASET
2. SMALL_DATASET

Execute the code in these sections by toggling the corresponding flag to true.

Select featurizers by toggling the following flags in main.py:
1. ZEROGRAM
2. W_SYNCTACTIC_CHUNK_WORD_FEATURIZER
3. W_SYNTACTIC_CHUNK_TAG_FEATURIZER
4. W_WORD_TAG_FEATURIZER
5. W_1_GRAM_TAG_FEATURIZER
6. W_1_GRAM_SYNCTACTIC_CHUNK_FEATURIZER
7. W_2_GRAM_TAG_FEATURIZER
8. W_2_GRAM_FORWARD_TAG_FEATURIZER
9. W_2_GRAM_SYNCTACTIC_CHUNK_FEATURIZER
10. W_2_GRAM_FORWARD_SYNCTACTIC_CHUNK_FEATURIZER
11. W_3_GRAM_TAG_FEATURIZER
12. W_3_GRAM_FORWARD_TAG_FEATURIZER
13. W_3_GRAM_AROUND_TAG_FEATURIZER
14. W_3_GRAM_SYNCTACTIC_CHUNK_FEATURIZER
15. W_3_GRAM_FORWARD_SYNCTACTIC_CHUNK_FEATURIZER
16. W_3_GRAM_AROUND_SYNCTACTIC_CHUNK_FEATURIZER
17. W_IS_NUMBER_FEATURIZER
18. W_IS_CAPITALIZED_FEATURIZER
19. W_IS_SPLIT_FEATURIZER

You can also toggle groups of featurizers using the following flags:
1. WORD_PROPERTIES - Toggles feature # 17, 18, 19 in list above
2. ONEGRAM - Toggles all OneGram featurizers (#2, #3, #4, #5, #6, #7 in list above)
3. BIGRAM - Toggles all BiGram featurizers (#7, #8, #9, #10 in list above)
4. TRIGRAM - Toggles TriGram featurizers (#11, #12, #13, #14, #15, #16 in list above)
5. ALL_FEATURES - Toggles all features.

Other Flags: 

1. EVAL_TRAIN: Toggles if train dataset is evaluated or not.
2. EVAL_TRAIN_SMALL: Toggles if train small dataset is evaluated or not.
3. FULL_DATASET: Toggles if full dataset is used or not.
4. SMALL_DATASET: Toggles if small dataset is used or not.


Libraries required:
1. numpy
2. tqdm
3. multiprocessing
4. joblib

----------------------------

Homework-2

Libraries required:
1. numpy
2. tqdm
3. multiprocessing
4. joblib


HW2: 

To run the assignment, execute `nohup python3.8 HW2/main.py > nohup.out &`

the file "main.py" is divided into 3 sections :
0. DEVEL (Alpha testing)
1. BIGRAM
    1.1 TRAIN
    1.2 DEV
    1.3 TEST
2. TRIGRAM
    2.1 TRAIN
    2.2 DEV
    2.3 TEST

Execute the code in these sections by toggling the corresponding flag to true.

----------------------------

Homework-1: 

To run the assignment, execute `nohup python3.8 HW1/main.py > nohup.out &`

the file "main.py" is divided into 3 sections :
1. TEST (Devo testing)
2. DEVELOP (Attempting the questions)
3. BRING_IT_TOGETHER (Attempting the bonus questions 4.1 and 4.2 along with executing on the test set)

Execute the code in these sections by toggling the corresponding flag to true.

