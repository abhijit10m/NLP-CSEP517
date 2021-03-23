from re import subn
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score, single_meteor_score
import nltk
import numpy as np

import os

from numpy.lib.shape_base import split

nltk.download('wordnet')
nltk.download('punkt')

with open('data/test_summaries.txt') as f:
    ground_truth = [line.strip() for line in f.readlines()]
    
with open('data/test_documents.txt') as f:
    documents = [line.strip() for line in f.readlines()]


####################
###### Q 1.2 #######
## Implement lead-5 and lead-10 baselines
## You will need to separate each sentence into
## words and then combine them back into a 
## sentence
##
## You should set lead_k to a list of strings
## where lead_k[i] is the a string containing 
## the first k words of documents[i] (above, 
## the documents we are summarizing)
##
## HINT: you will need to use the nltk word_tokenize 
## function
##
## HINT: you will be using the array ``documents''
##
## HINT: look into the .join() function for strings
##
## !!! Code goes here
lead = []
for k in [5, 10]:
    
    lead_k = [ " ".join(word_tokenize(word)[:k])  for word in documents ] ## !!! define this list as described above
    
    lead += [('lead-{}'.format(k),lead_k)]
##
##
####################
####################

for name, gens in lead:
    with open('generations/'+name,'w') as f:
        for gen in gens:
            f.write(gen + '\n')

gen_files = [f for f in os.listdir(path='generations/') if f.endswith('.txt')]

generations = []
for file in gen_files:
    with open('generations/'+file) as f:
        generations += [(file.replace('.txt',''), [line.strip() for line in f.readlines()])]



        



    
    
    
print('===========================')
print('Evaluating all models')
print('===========================')
        
for name, gens in generations + lead:
    
    bleu_scores = []
    meteor_scores = []
    for summ, gt in zip(gens, ground_truth):
        #######################################
        ##### Q 1.1 ###########################
        ## calculate the scores (meteor and bleu) 
        ## for the ground-truth summary ``gt''
        ## and the model summary ``summ''
        ##
        ## You will have to read the documentation for
        ## meteor_score and corpus_bleu to figure out what the input
        ## should look like
        ##
        ## HINTS:
        ##  - you should be using corpus bleu on a SINGLE example (gt, summ)
        ##    NOT the entire lists (gens, ground_truth)
        ##  - We want to give 50% weight to unigrams and 50% weight to bigrams 
        ##    for BLEU. See the documentation for how to do this
        ##  - meteor_score and corpus_bleu will NOT take inputs of the same format
        ##  - You will need to use the nltk word_tokenize function for one of these
        ##
        meteor_scores += [meteor_score([gt], summ)]

        bleu_scores += [corpus_bleu([[word_tokenize(gt)]], [word_tokenize(summ)], weights=[0.5,0.5])] 
        
        ## need to tokenize, then do corpus bleu
        ## input to corpus bleu should be 1) tokenized (i.e. a list of words) and 2) lowercase
        ## you do not need to remove punctuation
        ## this can be written in 2-4 lines of code
        #bleu_scores += [corpus_bleu()]
        #######################################
        #######################################
    print(name)
    print('METEOR: {}'.format(np.mean(meteor_scores)))
    print('BLEU: {}'.format(np.mean(bleu_scores)))


    
