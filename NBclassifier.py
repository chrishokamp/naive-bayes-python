#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from collections import Counter
import numpy as np
import itertools
import ipdb

''' 
Author: Chris Hokamp
NBclassifier can:
    - learn a model (Classifier.train)
    - if (model), classifier can classify a new instance
    - look at a list of tuples (feature, class)

'''

class NBclassifier:
    #create feature map on init (expect a vector of feature names)
    def __init__(self, featureVec, classVec, documents):
        self.featureMap = self.createFeatureMap(featureVec)
        self.indexMap = self.createIndexMap(featureVec)
        self.classMap = self.createClassMap(classVec)
        self.classIndexMap = {v:k for k, v in self.classMap.items()}
        self.pi, self.trainedMatrix = self.trainMultinomial(documents)

    # instances is a list of tuples with ({tok1: val}, class)
    # return a vector of class priors, and a matrix of c | f 
    def trainMultinomial(self, instances):
        # array is |classes| x |features|
        #featureCounts = np.ndarray(shape=(len(classMap.keys()), len(featureMap.keys())), dtype=int32)

        cSize = len(self.classMap.keys())
        fSize = len(self.featureMap.keys())

        # np.ones for add-one smoothing
        featureCounts = np.ones((cSize, fSize), dtype='float')
        classOccs = np.zeros((cSize))
        # sum feature occs accross all instances
        for docVec, c in instances:
            i = self.classMap[c]
            classOccs[i] += 1
            for feature in docVec.keys():
                j = self.featureMap[feature]    
                featureCounts[i,j] += 1
                
        print featureCounts
        # calculate priors on classes
        pi = classOccs/classOccs.sum() 
        #TEST
        print pi

        # calculate f | c
        # divide each row by the sum of that row + |V|
        featuresGivenClass = (featureCounts.T/featureCounts.sum(axis=1)).T
        print featuresGivenClass
        return pi, featuresGivenClass

    # return a list of (class, score) scores descending 
    # assume preprocessing of docVec
    def classify(self, docVec):
        scores = []
        for i, cVec in enumerate(self.trainedMatrix):
            score = np.log(self.pi[i]) 
            for word, count in docVec.items():
                print word, count
                #TODO; move to log space
                #TODO: if feature is not in dict, ignore it
                if word in self.featureMap:
                    j = self.featureMap[word]
                    timesOccs = np.log(self.trainedMatrix[i,j]) * count
                    score += timesOccs
            scores.append((self.classIndexMap[i], score))        
        scores.sort(key=lambda x: x[1])    
        desc = scores[::-1]
        return desc 

    #map classes to indexes  
    def createClassMap(self, classVec):
        return {v:k for k, v in dict(enumerate(classVec)).items()}

    #map feature names to indexes    
    def createFeatureMap(self, featureVec):
        return {v:k for k, v in dict(enumerate(featureVec)).items()}

    #map indexes to features    
    def createIndexMap(self, featureVec):
        return dict(enumerate(featureVec))

    def featureForIndex(self, index):
        return self.indexMap[index] 

    def indexForFeature(self, feature):
        return self.featureMap[feature] 

    # Use mutual information to prune to the top N features
    def getTopFeatures(self, N):    
        pass

