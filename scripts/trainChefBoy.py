# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 23:43:13 2017

@author: jmf
"""
from sklearn.decomposition import LatentDirichletAllocation as LDA
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import numpy as np
import os
import sys
import util

# create preprocessor
prep = util.Preprocessor()
prep.LDAsize = 25
prep.recipesFromFile()


# build dictionaries
prep.buildDict(topn=10000)

# train LDA model
#   - get spare matrices
trainInds, valInds = util.splitDocs(np.arange(len(prep.recipes)))
sparseMat = prep.toSparseMatrix(np.arange(len(prep.recipes)))

#   - fit_transform both train and val
lda = LDA(n_topics=prep.LDAsize)
prep.LDAvecs = lda.fit_transform(sparseMat)



# define model
model = util.defineModel(prep)

callbacks = [
    EarlyStopping(patience=5),
    ModelCheckpoint(filepath='../models/recipe.cnn', verbose=1, save_best_only=True),
    TensorBoard()
]

# create validation set
val = util.recipesToMatrices(prep, valInds)
Xval = val[:3]
yval = val[3:]

train = util.recipesToMatrices(prep, trainInds)
Xtrain = train[:3]
ytrain = train[3:]



# train model using generator
model.fit(Xtrain, ytrain, epochs=100, callbacks=callbacks,
          validation_data=(Xval,yval))