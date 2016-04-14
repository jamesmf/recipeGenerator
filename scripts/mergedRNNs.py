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
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

"""
script to train a word-level LSTM or GRU to predict the next word based on 
the previous words and characters, as well as the topic vector of the doc
"""


def sampleFromRecipes(recs):
    out     = []
    for recipe in recs:
        l           = max(0,len(recipe) - stripLen)
        startInd    = int((np.random.rand()<probStart)*l*np.random.rand())
        if np.random.rand() < probStart:            
            startInd    = int(min(max(len(recipe)-stripLen,0),recipe.find("name")))
        out.append(recipe[startInd:startInd+stripLen])
    return out

def recipesToChars(chars,char_indices,recipes):
    Xchar   = []
    Ychar   = []
    for recipe in recipes:
        char_snips  = []
        next_chars  = []
        for i in range(0, len(recipe) - maxlen, step):
            char_snips.append(recipe[i: i + maxlen])
            next_chars.append(recipe[i + maxlen])

        #iterate over snippets within one recipe
        xchar   = np.zeros((len(char_snips),maxlen,len(chars)))
        ychar   = np.zeros((len(char_snips),len(chars)))
        for i, charsnip in enumerate(char_snips):
            #turn each snippet into a one-hot encoded array of examples x time x output
                    
            #for each timestep in the snippet
            for t, char in enumerate(charsnip):

                #onehot character vector
                x_i     = np.zeros(len(char_indices))
                x_i[char_indices[char]] = 1

                xchar[i,t,:]    = x_i
            y_i     = np.zeros(len(chars))
            y_i[char_indices[next_chars[i]]] = 1
            
            
            ychar[i,:] = y_i
            
        Xchar.append(xchar)
        Ychar.append(ychar)
    
    XcharOut   = np.zeros((len(Ychar[0])*batchSize,maxlen,len(char_indices)))
    YcharOut   = np.zeros((len(Ychar[0])*batchSize,len(char_indices)))
    
    ind     = 0
    added   = 0
    no      = 0
    for i in range(0,len(Ychar[0])):
        for j in range(0,batchSize):
            if (j < len(Xchar)) and (i < len(Xchar[j])):
                XcharOut[ind,:,:] = Xchar[j][i,:]
                YcharOut[ind,:]   = Ychar[j][i]
                added+=1
            else:
                no+=1
            ind+=1

    return XcharOut,YcharOut


def recipesToWords(vocab,word_indices,recipes,recs):
    Xword   = []
    for recnumber, recipe in enumerate(recipes):
        ind1        = recipe.find(recs[recnumber])
        recipe      = recipe[0:ind1+len(recs[recnumber])]
        word_snips  = []
        next_chars  = []
        for i in range(0, len(recipe) - maxlen, step):
            
            word_snips.append(helper.wordTokenize(recipe[: i + maxlen])[:-1])
            next_chars.append(recipe[i + maxlen])

        #iterate over snippets within one recipe
        xword   = np.zeros((len(word_snips),maxlen))
        for i, wordsnip in enumerate(word_snips):
            #turn each snippet into a one-hot encoded array of examples x time x output
            if len(wordsnip) >= maxlen:
                diff        = 0
                wordsnip    = wordsnip[-maxlen:]
            else:
                diff        = maxlen - len(wordsnip)

            #for each timestep in the snippet
            for t, word in enumerate(wordsnip):
                tadjust         = t+diff
                if word in vocab:
                    xword[i,tadjust]= word_indices[word] 
            
        Xword.append(xword)
    
    XwordOut   = np.zeros((len(Xword[0])*batchSize,maxlen))
    
    ind     = 0
    added   = 0
    no      = 0
    for i in range(0,len(Xword[0])):
        for j in range(0,batchSize):
            if i < len(Xword[j]):
                XwordOut[ind,:] = Xword[j][i]
                added+=1
            else:
                no+=1
            ind+=1

    return XwordOut 



#in order to use stateful RNNs, we process in batches
batchSize   = 16


#read in the text file
path    = "../allrecipes.txt"
text    = open(path).read().lower()
recipes = [r+"$$$$" for r in text.split("$$$$")]
np.random.shuffle(recipes)
print("number of recipes:",len(recipes))

#define the character vocabulary
chars = list(set(text))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#define the word vocabulary
word_thr= 2
toks    = helper.wordTokenize(text)
counts  = Counter(toks)
vocab   = [x[0] for x in counts.most_common() if x[1] > word_thr]
vocSize = len(vocab)
print('corpus length (characters):', len(text))
print('corpus length (tokens)', )
print('vocab size:', vocSize)

word_indices = dict((c, i+1) for i, c in enumerate(vocab))
indices_word = dict((i+1, c) for i, c in enumerate(vocab))

maxlen   = 30
step     = 1
stripLen = 100
probStart= 0.1


#print(contextVecs[:3])
#stop=raw_input("done transforming docs")

#define model
if len(sys.argv) > 1:
    model   = helper.loadThatModel(sys.argv[1])
else:
    charModel = Sequential()
    charModel.add(LSTM(512, return_sequences=True,batch_input_shape=(batchSize,maxlen, len(chars)),stateful=True))
    charModel.add(Dropout(.2))
    #charModel.add(TimeDistributedDense(128))
    
    wordModel = Sequential()
    wordModel.add(Embedding(vocSize+1, 512, input_length=maxlen,batch_input_shape=(batchSize,maxlen)))
    wordModel.add(LSTM(512, return_sequences=True,stateful=True))    
    wordModel.add(Dropout(.2))
    wordModel.add(LSTM(512, return_sequences=False,stateful=True))
    wordModel.add(RepeatVector(maxlen))
    
    
    model = Sequential()
    model.add(Merge([charModel, wordModel], mode='concat', concat_axis=-1))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dense(128))
    model.add(Dropout(.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#create a batch of recipes
ind         = 0
epochLoss   = []
while True: 
    if ind + batchSize > len(recipes):
        print()
        print("epoch complete:", np.mean(epochLoss,axis=0))
        epochLoss   = []
        ind         = 0
        np.random.shuffle(recipes)
    callbacks   = []
    recs        = recipes[ind:ind+batchSize]
    recs        = sorted(recs,key=lambda x: len(x),reverse=True)
    recSamples  = sampleFromRecipes(recs)
    Xchar,y     = recipesToChars(chars,char_indices,recSamples)
    Xword       = recipesToWords(vocab,word_indices,recs,recSamples)
    
    for i in range(0,Xchar.shape[0]/batchSize):
        xcharbat    = Xchar[i*batchSize:(i+1)*batchSize,:,:]
        xwordbat    = Xword[i*batchSize:(i+1)*batchSize,:]
        ybat        = y[i*batchSize:(i+1)*batchSize,:] 
        loss    = model.train_on_batch([xcharbat,xwordbat], ybat)
        if not np.isnan(loss[0]):
            callbacks.append(loss[0])

    print(ind,"/",len(recipes)," - loss:",np.mean(callbacks,axis=0),end="\r")
    sys.stdout.flush()
    epochLoss.append(np.mean(callbacks,axis=0))
    model.reset_states()
    ind     = ind+batchSize
    


    if (ind / batchSize) % 50 == 0:
        print()
        jsonstring  = model.to_json()
        with open("../models/recipeRNNMergedStateful2.json",'wb') as f:
            f.write(jsonstring)
        model.save_weights("../models/recipeRNNMergedStateful2.h5",overwrite=True)