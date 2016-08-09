from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense,RepeatVector,Merge
from keras.layers.recurrent import GRU,LSTM
from keras.models import model_from_json

import helper
import numpy as np
import re
from collections import Counter
import sys
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

"""
script to train a word-level LSTM or GRU to predict the next word based on 
the previous words and characters, as well as the topic vector of the doc
"""

#read in the text file
path = "../allrecipes.txt"
text = open(path).read().lower()
#convert to document list, padded with " " in front and "$$$$" to end
recipes = [" "*10+r+"$$$$" for r in text.split("$$$$") if (len(r) > 100 and len(r)<600)]
recipes = [r for r in recipes if not helper.excludeRecipe(r)]



#define a ModelHelper object to store vocab, preprocessing functions, etc
mh     = helper.ModelHelper(text)
mh.fitVocab(text,tf_threshold=10)


np.random.shuffle(recipes)
print("number of recipes:",len(recipes))


batchSize= 32
maxlen = 10
maxWord = 40
step = 3
#stripLen = 200
#probStart= 0.1
numComps = 25
vecSize  = numComps+1

mh.add("maxlen",maxlen)
mh.add("maxWord",maxWord)
mh.add("vecSize",vecSize)

ingNums = helper.getNumberIngredients(recipes)
names = helper.getNames(recipes)
#pca = PCA(n_components=numComps)
lda = LatentDirichletAllocation(n_topics=numComps,n_jobs=1)
#svd = TruncatedSVD(n_components=numComps)
cv = CountVectorizer(min_df=mh.tf_threshold)

print("countVectorizing recipes")
cvRecs   = cv.fit_transform(recipes)
print("recipes countVectorized")
#vecs  = pca.fit_transform(cvRecs)
print("Generating recipe-level stats (SVD or LDA)")
vecs  = lda.fit_transform(cvRecs)
#vecs   = svd.fit_transform(cvRecs)
print("Done")

contextVecs = [np.append(vecs[i],ingNums[i]) for i in range(0,len(recipes))]
contextVecs = np.array(contextVecs)
contextVecs = contextVecs - np.mean(contextVecs,axis=0)
contextVecs = contextVecs / (np.std(contextVecs,axis=0)+0.00000001)


#these will be useful for sampling randomly as a seed
mh.add("contextVectors",contextVecs)
mh.add("recipeNames",names)
mh.add("recipes",recipes)

mh.save("../models/recipeRNN3noState")


#define model
if len(sys.argv) > 1:
    model = helper.loadThatModel(sys.argv[1])
    mh = helper.loadHelper(sys.argv[1])
    contextVecs = mh.get("contextVectors")
    recipes = mh.get("recipes")
else:
    charModel = Sequential()
    charModel.add(LSTM(128, return_sequences=True,input_shape=(maxlen, mh.numChars)))
    charModel.add(Dropout(.2))
    #charModel.add(TimeDistributedDense(128))
    
    wordModel = Sequential()
    wordModel.add(Embedding(mh.vocSize, 512, input_length=maxWord))
    wordModel.add(Dropout(0.2))
    wordModel.add(LSTM(1024, return_sequences=True))    
    wordModel.add(Dropout(.2))
    wordModel.add(LSTM(1024, return_sequences=False))
    wordModel.add(RepeatVector(maxlen))

    #print(np.sum(np.ndarray.flatten(np.array([wordModel.layers[ind].get_weights() for ind in range(0,len(wordModel.layers))]))))
    if False:
        wM  = helper.loadThatModel("../models/pretrainedWord")
        for n,layer in enumerate(wordModel.layers[:-1]):
            wordModel.layers[n].set_weights(wM.layers[n].get_weights())
    #print(np.sum(np.ndarray.flatten(np.array([wordModel.layers[ind].get_weights() for ind in range(0,len(wordModel.layers))]))))
    
    
    contextModel = Sequential()
    contextModel.add(Dense(256,input_shape=(vecSize,)))
    #contextModel.add(Dense(512))
    contextModel.add(Activation('relu'))
    contextModel.add(RepeatVector(maxlen))
    
    model = Sequential()
    model.add(Merge([charModel, wordModel, contextModel], mode='concat', concat_axis=-1))
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(1024, return_sequences=False))
    model.add(Dropout(.2))
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dense(mh.numChars))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#create a batch of recipes
ind         = 0
print("begin training")
while True: 
    if ind + batchSize > len(recipes):
        print("new epoch")
        ind         = 0
        recipes, contextVecs = helper.sameShuffle(recipes,contextVecs)
        contextVecs = np.array(contextVecs)
    Xcontext = contextVecs[ind:ind+batchSize]
    recs = recipes[ind:ind+batchSize]
    recs = helper.shuffleIngredients(recs)
    
    Xcharacter, Xword, Xcontext, y = helper.getCharAndWordNoState(recs,Xcontext,maxlen,maxWord,mh.char_indices,mh.word_indices,step=step)
   

    loss = model.fit([Xcharacter, Xword, Xcontext], y,nb_epoch=1,batch_size=batchSize)

    ind = ind+batchSize

    if (ind / batchSize) % 10 == 0:
        print(ind)
        jsonstring  = model.to_json()
        with open("../models/recipeRNN3noState.json",'wb') as f:
            f.write(jsonstring)
        model.save_weights("../models/recipeRNN3noState.h5",overwrite=True)