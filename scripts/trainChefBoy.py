# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 23:43:13 2017

@author: jmf
"""
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
import os
import sys
import util

# create preprocessor
prep = util.Preprocessor()
prep.recipesFromFile()

# build dictionaries
prep.buildDict(topn=5000)

# train LDA model
#   - get spare matrices
trainInds, valInds = util.splitDocs(np.arange(len(prep.recipes)))
sparseMat = prep.toSparseMatrix(np.arange(len(prep.recipes)))

#   - fit_transform both train and val
lda = LDA(n_components=25)
prep.LDAvecs = lda.fit_transform(sparseMat)

# define RNN


# create validation set


# train model using generator