import os
import logging
import multiprocessing.dummy as mp
import numpy as np
from datetime import datetime
from itertools import product
from collections import defaultdict
from tqdm.auto import tqdm

format = "%(asctime)s: %(message)s"
logfileName = os.path.join("logs", "HW3", str(os.getpid())+"_"+datetime.now().strftime('progress_log_%H_%M_%d_%m_%Y.log') )
logging.basicConfig(filename=logfileName, filemode='a', format=format, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger("__HW3__")

def getNGrams(words, n):
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

def curateLine(line, start, end):
    line = np.insert(line,0,values=start, axis=0)
    line = np.append(line,values=end, axis=0)
    return line

class ConllFeaturizer(object):
    def __init__(self, train_corpus, start_symbol="<s>", end_symbol="<e>"):
        self.tags=set()
        self.words=set()
        self.syntactic_chunk_tags=set()
        self.named_entity_recognition_tags=set()
        self.totalWords = 0
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.max_n = 3
        self.start = [[self.start_symbol, self.start_symbol, self.start_symbol, 'O'] for i in range(self.max_n - 1)]
        self.end = [[self.end_symbol, self.end_symbol, self.end_symbol, 'O'] for i in range(self.max_n - 1)]
        self.word_featurizers = []

        # Feature => Index
        self.tag_ngrams = [ [] for i in range(self.max_n)]
        self.syntactic_chunk_ngrams = [ [] for i in range(self.max_n)]
        self.word_tag = []
        self.syntactic_chunk_tag = []
        self.syntactic_chunk_word = []

        # Index => Feature
        self.tag_ngrams_index = [ [] for i in range(self.max_n)]
        self.syntactic_chunk_ngrams_index = [ [] for i in range(self.max_n)]
        self.word_tag_index = []
        self.syntactic_chunk_tag_index = []
        self.syntactic_chunk_word_index = []

        for i in range(len(train_corpus)):
            line = curateLine(train_corpus[i], self.start, self.end)
            self.words.update(line[:,0])
            self.tags.update(line[:,1])
            self.syntactic_chunk_tags.update(line[:,2])
            self.named_entity_recognition_tags.update(line[:,3])

        self.words = sorted(self.words)
        self.tags = sorted(self.tags)
        self.syntactic_chunk_tags = sorted(self.syntactic_chunk_tags)
        self.named_entity_recognition_tags = dict(((ner, index) for (index, ner) in enumerate(sorted(self.named_entity_recognition_tags))))
        self.named_entity_recognition_tags_index = dict(((index, ner) for (index, ner) in enumerate(sorted(self.named_entity_recognition_tags))))

        for i in reversed(range(self.max_n)):
                self.tag_ngrams[i] = [" ".join(ngram) for ngram in product(self.tags, repeat=i+1)]
                self.tag_ngrams_index[i] = dict(((index, ngram) for (index, ngram) in enumerate(sorted(self.tag_ngrams[i]))))
                self.tag_ngrams[i] = dict(((ngram, index) for (index, ngram) in enumerate(sorted(self.tag_ngrams[i]))))
                self.syntactic_chunk_ngrams[i] = [" ".join(ngram) for ngram in product(self.syntactic_chunk_tags, repeat=i+1)]
                self.syntactic_chunk_ngrams_index[i] = dict(((index, ngram) for (index, ngram) in enumerate(sorted(self.syntactic_chunk_ngrams[i]))))
                self.syntactic_chunk_ngrams[i] = dict(((ngram, index) for (index, ngram) in enumerate(sorted(self.syntactic_chunk_ngrams[i]))))

        self.word_tag = dict(((word_tag, index) for (index, word_tag) in enumerate(sorted(product(self.words, self.tags)))))
        self.word_tag_index = dict(((index, word_tag) for (index, word_tag) in enumerate(sorted(product(self.words, self.tags)))))

        self.syntactic_chunk_tag_index = dict(((index, syntactic_chunk_tag) for (index, syntactic_chunk_tag) in enumerate(sorted(product(self.syntactic_chunk_tags, self.tags)))))
        self.syntactic_chunk_tag = dict(((syntactic_chunk_tag, index) for (index, syntactic_chunk_tag) in enumerate(sorted(product(self.syntactic_chunk_tags, self.tags)))))

        self.syntactic_chunk_word_index = dict(((index, syntactic_chunk_word) for (index, syntactic_chunk_word) in enumerate(sorted(product(self.syntactic_chunk_tags, self.words)))))
        self.syntactic_chunk_word = dict(((syntactic_chunk_word, index) for (index, syntactic_chunk_word) in enumerate(sorted(product(self.syntactic_chunk_tags, self.words)))))


        logger.debug("tags: %s", self.tags)
        logger.debug("words: %s", self.words)
        logger.debug("syntactic_chunk_tags: %s", self.syntactic_chunk_tags)
        logger.info("named_entity_recognition_tags: %s", self.named_entity_recognition_tags)

        logger.debug("tag_ngrams : %s", self.tag_ngrams)
        logger.debug("tag_ngrams_index : %s", self.tag_ngrams_index)
        logger.debug("syntactic_chunk_ngrams : %s", self.syntactic_chunk_ngrams)
        logger.debug("syntactic_chunk_ngrams_index : %s", self.syntactic_chunk_ngrams_index)
        logger.debug("word_tag : %s", self.word_tag)
        logger.debug("word_tag_index : %s", self.word_tag_index)
        logger.debug("syntactic_chunk_tag : %s", self.syntactic_chunk_tag)
        logger.debug("syntactic_chunk_tag_index : %s", self.syntactic_chunk_tag_index)
        logger.debug("syntactic_chunk_word : %s", self.syntactic_chunk_word)
        logger.debug("syntactic_chunk_word_index : %s", self.syntactic_chunk_word_index)

    def w_synctactic_chunk_word_featurizer(self, line, pos):
        features = defaultdict(int)
        features[self.syntactic_chunk_word[(line[pos, 2], line[pos, 0])]] = 1
        return features

    def w_word_tag_featurizer(self, line, pos):
        features = defaultdict(int)
        features[self.word_tag[(line[pos, 0], line[pos, 1])]] = 1
        return features

    def w_syntactic_chunk_tag_featurizer(self, line, pos):
        features = defaultdict(int)
        features[self.syntactic_chunk_tag[(line[pos, 2], line[pos, 1])]] = 1
        return features

    def w_1_gram_tag_featurizer(self, line, pos):
        features = defaultdict(int)

        features[self.tag_ngrams[0][line[pos, 1]]] = 1
        return features

    def w_2_gram_tag_featurizer(self, line, pos):
        features = defaultdict(int)
        ngram = " ".join(line[pos-1:pos+1, 1])

        features[self.tag_ngrams[1][ngram]] = 1
        return features

    def w_3_gram_tag_featurizer(self, line, pos):
        features = defaultdict(int)
        ngram = " ".join(line[pos-2:pos+1, 1])

        features[self.tag_ngrams[2][ngram]] = 1
        return features

    def w_2_gram_forward_tag_featurizer(self, line, pos):
        features = defaultdict(int)
        ngram = " ".join(line[pos:pos+1+1, 1])
        
        features[self.tag_ngrams[1][ngram]] = 1
        return features

    def w_3_gram_forward_tag_featurizer(self, line, pos):
        features = defaultdict(int)
        ngram = " ".join(line[pos:pos+2+1, 1])

        features[self.tag_ngrams[2][ngram]] = 1
        return features

    def w_3_gram_around_tag_featurizer(self, line, pos):
        features = defaultdict(int)
        ngram = " ".join(line[pos-1:pos+2, 1])
        
        features[self.tag_ngrams[2][ngram]] = 1
        return features

    def w_1_gram_synctactic_chunk_featurizer(self, line, pos):
        features = defaultdict(int)

        features[self.syntactic_chunk_ngrams[0][line[pos, 2]]] = 1
        return features

    def w_2_gram_synctactic_chunk_featurizer(self, line, pos):
        features = defaultdict(int)
        ngram = " ".join(line[pos-1:pos+1, 2])
    
        features[self.syntactic_chunk_ngrams[1][ngram]] = 1
        return features

    def w_3_gram_synctactic_chunk_featurizer(self, line, pos):
        features = defaultdict(int)
        ngram = " ".join(line[pos-2:pos+1, 2])

        features[self.syntactic_chunk_ngrams[2][ngram]] = 1
        return features

    def w_2_gram_forward_synctactic_chunk_featurizer(self, line, pos):
        features = defaultdict(int)
        ngram = " ".join(line[pos:pos+1+1, 2])

        features[self.syntactic_chunk_ngrams[1][ngram]] = 1
        return features

    def w_3_gram_forward_synctactic_chunk_featurizer(self, line, pos):
        features = defaultdict(int)
        ngram = " ".join(line[pos:pos+2+1, 2])

        features[self.syntactic_chunk_ngrams[2][ngram]] = 1
        return features

    def w_3_gram_around_synctactic_chunk_featurizer(self, line, pos):
        features = defaultdict(int)
        ngram = " ".join(line[pos-1:pos+2, 2])

        features[self.syntactic_chunk_ngrams[2][ngram]] = 1
        return features

    def w_is_number_featurizer(self, line, pos):
        features = defaultdict(int)

        features["is_number"] = 1 if line[pos, 0].isnumeric() == True else 0
        features["is_not_number"] = 0 if line[pos, 0].isnumeric() == True else 1
        return features

    def w_is_capitalized_featurizer(self, line, pos):
        features = defaultdict(int)

        features["is_capitalized"] = 1 if line[pos, 0].isupper() == True else 0
        features["is_not_capitalized"] = 0 if line[pos, 0].isupper() == True else 1
        return features

    def w_is_split_featurizer(self, line, pos):
        features = defaultdict(int)
        features["is_split"] = 1 if "-" in line[pos, 0] else 0
        features["is_not_split"] = 0 if "-" in line[pos, 0] else 1
        return features

    def featurize_word(self, line, pos):
        x = [f(line, pos) for f,featureCount in self.word_featurizers]
        logger.debug("pos: %d, line: %s, features: %s", pos, line, x)
        logger.debug("__stub__ConllFeaturizer__featurize_word")
        assert len(x) == len(self.word_featurizers)
        return x

    def featurize_line(self, line):
        x = []
        for i in range(2, len(line) - 2):
            x.append(np.asarray(self.featurize_word(line, i)))
        x = np.asarray(x)
        logger.debug("line: %s, features: %s", line[:,0],  x)
        logger.debug("__stub__ConllFeaturizer__featurize_line")
        assert len(x) == len(line)-4
        self.pbar.update(1)
        return x

    def featurizeTrain(self, corpus):
        return self.featurize(corpus)

    def featurize(self, corpus):
        curated_corpus = []
        featureCountPerWord = 0
        self.pbar = tqdm(total=len(corpus), desc="Featurizer")

        for i in range(len(corpus)):
            curated_corpus.append(curateLine(corpus[i], self.start, self.end))

        with mp.Pool() as p:
            x = p.map(self.featurize_line, curated_corpus)
        p.close()
        p.join()
        self.pbar.close()
        featureCountPerWord = [featureCount for f,featureCount in self.word_featurizers]

        return x, featureCountPerWord