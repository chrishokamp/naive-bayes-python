#!/usr/bin/env python
# -*- coding: utf-8 -*-

import VectorBuilder as vb
import NBclassifier as nb
import preprocessAndCount as preprocess

#setup
docs = [("this is test document 1.", "test1"), ("this is test document 1.", "test1"), ("this is test document 2.", "test2")]

corpus = vb.WordVectorBuilder("english", docs)

nbClassifier = nb.NBclassifier(corpus.features, corpus.classes, corpus.vectorCollection)

# training -- TODO: test this
#nbClassifier.trainMultinomial(corpus.vectorCollection)

# classify new instance
test1 = "test 2 test 2"
testVec = preprocess.getCountsVector(test1)
scores = nbClassifier.classify(testVec)
print scores

# Test counting features in NBclassifier.py#trainSimple
#print featureCounts
#for i, count in enumerate(featureCounts[0,]):
#    print "feature: %s, count: %i" % (self.featureForIndex(i),count)

#classification

