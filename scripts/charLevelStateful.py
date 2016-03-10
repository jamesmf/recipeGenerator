from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense,RepeatVector,Merge
from keras.layers.recurrent import GRU, LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import re
import operator
from collections import Counter
import sys
from keras.models import model_from_json

"""
script to train a char-level LSTM or GRU to predict the next word based on 
the previous chars, statefully
"""

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def checkExample(inp,dic):
    if inp.shape[0] == len(dic):
        inp     = np.reshape(inp,(1,inp.shape[0]))
    print(inp.shape)
    num     = inp.shape[0]
    for temp in range(0,num):
        #print(i    np[temp])
        #print(np.sum(inp[temp]))
        if np.sum(inp[temp]) > 0:
            #print(list(inp[temp]))
            #print(list(inp[temp]).index(1))
            print(dic[list(inp[temp]).index(1)],end="")
    print()

def sampleFromRecipes(recs):
    out     = []
    for recipe in recs:
        l           = max(0,len(recipe) - stripLen)
        startInd    = int((np.random.rand()<probStart)*l*np.random.rand())
        out.append(recipe[startInd:startInd+stripLen])
    return out

def processRecipes(chars,char_indices,recipes,maxlen,step):
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

def loadThatModel(folder):
    with open(folder+".json",'rb') as f:
        json_string     = f.read()
    model = model_from_json(json_string)
    model.load_weights(folder+".h5")
    return model




#read in the text file
path    = "../allrecipes.txt"
text    = open(path).read().lower()
recipes = [r+"$$$$" for r in text.split("$$$$")]
np.random.shuffle(recipes)
#recipes = sorted(recipes,key=lambda x: len(x),reverse=True)
print("number of recipes:",len(recipes))

#define the character vocabulary
chars = list(set(text))
print(chars)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#define the word vocabulary
stripLen = 500
probStart= 0.25
maxlen   = 20
step     = 1
batchSize= 16

if len(sys.argv) > 1:
    model   = loadThatModel(sys.argv[1])
else:
    #define model
    print("compiling model")
    model = Sequential()
    model.add(LSTM(512, return_sequences=True,batch_input_shape=(batchSize,maxlen, len(chars)),stateful=True))
    model.add(Dropout(.2))
    model.add(LSTM(512,return_sequences=True,stateful=True))
    model.add(Dropout(.2))
    model.add(LSTM(256,return_sequences=False,stateful=True))
    model.add(Dropout(.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print("model compiled")

ind     = 0
callbacks   = []
epochLoss   = []
while True: 
    callbacks   = []
    recs        = recipes[ind:ind+batchSize]
    recs        = sampleFromRecipes(recs)
    Xchar,y     = processRecipes(chars,char_indices,recs,maxlen,step)
    for i in range(0,Xchar.shape[0]/batchSize):
        xbat    = Xchar[i*batchSize:(i+1)*batchSize,:,:]
        ybat    = y[i*batchSize:(i+1)*batchSize,:]
        #print(checkExample(xbat[0],indices_char))
        #print(checkExample(ybat[0],indices_char))
        loss    = model.train_on_batch(xbat, ybat)
        if not np.isnan(loss[0]):
            callbacks.append(loss[0])

    print(ind,"/",len(recipes)," - loss:",np.mean(callbacks,axis=0),end="\r")
    sys.stdout.flush()
    epochLoss.append(np.mean(callbacks,axis=0))
    model.reset_states()
    ind     = ind+batchSize
    
    if ind > len(recipes):
        print()
        print("epoch complete:", np.mean(epochLoss,axis=0))
        epochLoss   = []
        ind         = 0
        #np.random.shuffle(recipes)
        #getNewRecipe(vocab,word_indices,chars,char_indices,maxlen,step,batchSize)
        jsonstring  = model.to_json()
        with open("../recipeLSTMcharState.json",'wb') as f:
            f.write(jsonstring)
        model.save_weights("../recipeLSTMcharState.h5",overwrite=True)