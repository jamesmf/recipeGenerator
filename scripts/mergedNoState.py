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
path    = "../allrecipes.txt"
text    = open(path).read().lower()
#convert to document list, padded with " " in front and "$$$$" to end
recipes = [" "*10+r+"$$$$" for r in text.split("$$$$") if (len(r) > 100 and len(r)<600)]



#define a ModelHelper object to store vocab, preprocessing functions, etc
mh     = helper.ModelHelper(text)
mh.fitVocab(text,tf_threshold=2)


np.random.shuffle(recipes)
print("number of recipes:",len(recipes))

#define the character vocabulary
#chars = list(set(text))
#print('total chars:', len(chars))
#char_indices = dict((c, i) for i, c in enumerate(chars))
#indices_char = dict((i, c) for i, c in enumerate(chars))

#define the word vocabulary
#word_thr= 2
#toks    = helper.wordTokenize(text)
#counts  = Counter(toks)
#vocab   = [x[0] for x in counts.most_common() if x[1] > word_thr]
#vocSize = len(vocab)
#print(vocab[:10])
#stop=raw_input("")
#print('corpus length (characters):', len(text))
#print('corpus length (tokens)', )
#print('vocab size:', vocSize)
#
#word_indices = dict((c, i+1) for i, c in enumerate(vocab))
#indices_word = dict((i+1, c) for i, c in enumerate(vocab))

batchSize= 32
maxlen   = 20
maxWord  = 20
step     = 3
#stripLen = 200
#probStart= 0.1
numComps = 25
vecSize  = numComps+1

mh.add("maxlen",maxlen)
mh.add("maxWord",maxWord)
mh.add("vecSize",vecSize)

ingNums = helper.getNumberIngredients(recipes)
names   = helper.getNames(recipes)
#pca      = PCA(n_components=numComps)
#lda      = LatentDirichletAllocation(n_topics=numComps,n_jobs=1)
svd     = TruncatedSVD(n_components=numComps)
cv      = CountVectorizer(min_df=mh.tf_threshold)

print("countVectorizing recipes")
cvRecs   = cv.fit_transform(recipes)
print("recipes countVectorized")
#pcaVecs  = pca.fit_transform(cvRecs)
print("Generating recipe-level stats (SVD or LDA)")
#ldaVecs  = lda.fit_transform(cvRecs)
svdVecs   = svd.fit_transform(cvRecs)
print("Done")

contextVecs     = [np.append(svdVecs[i],ingNums[i]) for i in range(0,len(recipes))]
contextVecs     = np.array(contextVecs)
contextVecs     = contextVecs - np.mean(contextVecs,axis=0)
contextVecs     = contextVecs / (np.std(contextVecs,axis=0)+0.00000001)
#print(contextVecs[:3])
#stop=raw_input("done transforming docs")

#these will be useful for sampling randomly as a seed
mh.add("contextVectors",contextVecs)
mh.add("recipeNames",names)

mh.save("../models/recipeRNN3noState")

#define how we go from document 


#define model
if len(sys.argv) > 1:
    model   = helper.loadThatModel(sys.argv[1])
    mh      = helper.loadHelper(sys.argv[1])
else:
    charModel = Sequential()
    charModel.add(LSTM(128, return_sequences=True,input_shape=(maxlen, mh.numChars)))
    charModel.add(Dropout(.2))
    #charModel.add(TimeDistributedDense(128))
    
    wordModel = Sequential()
    wordModel.add(Embedding(mh.vocSize+1, 256, input_length=maxWord))
    wordModel.add(LSTM(512, return_sequences=True))    
    wordModel.add(Dropout(.2))
    wordModel.add(LSTM(512, return_sequences=False))
    wordModel.add(RepeatVector(maxlen))
    
    contextModel    = Sequential()
    contextModel.add(Dense(256,input_shape=(vecSize,)))
    #contextModel.add(Dense(512))
    contextModel.add(Activation('relu'))
    contextModel.add(RepeatVector(maxlen))
    
    model = Sequential()
    model.add(Merge([charModel, wordModel, contextModel], mode='concat', concat_axis=-1))
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(.2))
    model.add(Dense(mh.numChars))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#create a batch of recipes
ind         = 0
print("begin training")
while True: 
    if ind + batchSize > len(recipes):
        ind         = 0
        np.random.shuffle(recipes)
    Xcontext    = contextVecs[ind:ind+batchSize]
    recs        = recipes[ind:ind+batchSize]
    recs        = helper.shuffleIngredients(recs)
    
    Xcharacter, Xword, Xcontext, y  = helper.getCharAndWordNoState(recs,Xcontext,maxlen,maxWord,mh.char_indices,mh.word_indices,step=step)
   

    loss    = model.fit([Xcharacter, Xword, Xcontext], y,nb_epoch=1,batch_size=batchSize)

    ind     = ind+batchSize

    if (ind / batchSize) % 10 == 0:
        print()
        jsonstring  = model.to_json()
        with open("../models/recipeRNN3noState.json",'wb') as f:
            f.write(jsonstring)
        model.save_weights("../models/recipeRNN3noState.h5",overwrite=True)