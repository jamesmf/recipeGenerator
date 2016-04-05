from __future__ import print_function
import numpy as np
import re
from collections import Counter
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense,RepeatVector,Merge
from keras.layers.recurrent import GRU,LSTM
from keras.models import model_from_json
import helper

"""
script to train a char-level LSTM or GRU to predict the next word based on 
the previous words, previous chars, and topic vector
"""

#load our ModelHelper, pickled when training
mh      = helper.loadHelper("../models/recipeRNN3noState")

temp    = mh.word_indices

maxlen      = mh.get("maxlen")
maxWord     = mh.get("maxWord")
vecSize     = mh.get("vecSize")
batchSize   = 4

names       = mh.get("recipeNames")
conVecs     = mh.get("contextVectors")

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

model.load_weights("../models/recipeRNN3noState.h5")

#model   = helper.loadThatModel("../models/recipeRNN3noState")




while True:
    randomInd   = int(np.floor(np.random.rand()*len(names)))
    name        = " "*maxlen + names[randomInd]

    #name        = name[-maxlen-1:]
    contextVec  = conVecs[randomInd]*np.ones((batchSize,conVecs.shape[1]))
    #print(contextVec.shape)
    
    recipes     = [name for i in range(0,batchSize)]
    div         = np.linspace(0.2,1.,batchSize)
    startLen    = len(recipes[0])
    
    #create a batch of recipes
    ind         = 0
    epochLoss   = []
    for cInd in range(400): 
#        if (cInd+maxlen) < startLen:
#            recs    = [r[cInd:cInd+maxlen] for r in recipes]
#        else:
#            recs  = [r[-maxlen:] for r in recipes]
        
        #print(maxlen, maxWord,len(mh.char_indices),len(mh.word_indices))
        Xchar,Xword,Xcon,dummy    = helper.getCharAndWordNoState(recipes,contextVec,maxlen,maxWord,mh.char_indices,mh.word_indices,step=1,predict=True)
        newLength   = (Xchar.shape[0])/4        
        
        inds        = [(newLength*(divind+1))-1 for divind in range(0,batchSize)]
        #helper.checkExampleWords(Xword[inds[1]],mh.vocab)

        Xchar   = Xchar[inds]
        Xword   = Xword[inds]
        Xcon    = Xcon[inds]        
        
        

        preds       = model.predict_on_batch([Xchar,Xword,Xcon])[0]
        for d,pred in enumerate(preds):
            #print(d,pred)
            next_index  = helper.sample(pred, div[d])
            next_char   = mh.indices_char[next_index]
            
    
            if recipes[d][-1]  != "$":
                recipes[d] += next_char
        
        dropIt  = [(r[-1] == '$')*1. for r in recipes]
        if int(np.sum(dropIt)) == batchSize:
            break
    with open("../somerecipesfriend.txt",'a') as f:
        for d,rec in enumerate(recipes):
            print("with diversity:",div[d],"\n\n",rec,'\n\n')
            f.write("with diversity:"+str(div[d])+"\n\n"+rec+'\n\n')