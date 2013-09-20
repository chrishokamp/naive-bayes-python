#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk as nltk
from collections import Counter
import itertools

''' 
Author: Chris Hokamp
VectorBuilder uses NLTK to take a document (a string), and return a vector of word counts
    - look at a list of tuples (feature, class)

'''

# arg 'documents' is assumed to be a list of tuples, where each tuple is of the form (string, classLabel)
class WordVectorBuilder:
    def __init__(self, lang, documents):
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.word_tokenize = nltk.tokenize.word_tokenize
        self.stemmer = nltk.stem.snowball.SnowballStemmer(lang)
        self.vectorCollection = self.buildCollection(documents)
        self.features, self.classes = self.uniqueFeaturesAndClasses()

    #returns a list of tuples with the format ({docVector}, class)
    def buildCollection(self, documents):
        return [(self.getCountsVector(doc), c) for doc, c in documents]     
    #return the set of all unique tokens
    def uniqueFeaturesAndClasses(self):
        l = []
        c = []
        for vec, cName in self.vectorCollection: #very messy! 
            l = l + vec.keys()
            c.append(cName) 
        return list(set(l)), list(set(c))

    def getCountsVector(self, text):
        stemmed = self.preprocess(text)
        countsVector = Counter(stemmed)  
        return countsVector
   
    def preprocess(self, text):
        #convert text to unicode if it's not?
        sents = self.sentence_tokenizer.tokenize(text)
        toks = [self.word_tokenize(sen) for sen in self.sentence_tokenizer.tokenize(text)]
        stemmed = []
        for sen in toks:
            for tok in sen:
                try:
                    #print tok
                    stemmed.append(self.stemmer.stem(tok))
                except UnicodeDecodeError:
                    pass
        #This was throwing UnicodeDecodeError
        #stemmed = [self.stemmer.stem(tok) for sen in toks for tok in sen]
        return stemmed
