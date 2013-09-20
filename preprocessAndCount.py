#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk as nltk
from collections import Counter
import itertools

''' 
Author: Chris Hokamp
    - preprocessing utils for vectors of words

'''

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenize = nltk.tokenize.word_tokenize
stemmer = nltk.stem.snowball.SnowballStemmer("english")

def getCountsVector(text):
    stemmed = preprocess(text)
    countsVector = Counter(stemmed)  
    return countsVector
   
def preprocess(text):
    sents = sentence_tokenizer.tokenize(text)
    toks = [word_tokenize(sen) for sen in sentence_tokenizer.tokenize(text)]
    stemmed = []
    for sen in toks:
        for tok in sen:
            try:
                #print tok
                stemmed.append(self.stemmer.stem(tok))
            except UnicodeDecodeError:
                pass
    return stemmed
