from __future__ import print_function
#from keras.models import Sequential
#from keras.layers.embeddings import Embedding
#from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense,RepeatVector,Merge
#from keras.layers.recurrent import GRU,LSTM
#from keras.datasets.data_utils import get_file
import numpy as np
import re
import operator
from collections import Counter
import sys
from keras.models import model_from_json

"""
script to train a word-level LSTM or GRU to predict the next word based on 
the previous hist words
"""

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def checkExample(inp,dic):
    if inp.shape[0] == len(dic):
        inp     = np.reshape(inp,(1,inp.shape[0]))
    num     = inp.shape[0]
    for temp in range(0,num):
        if np.sum(inp[temp]) > 0:
            print(dic[list(inp[temp]).index(1)],end="")
            
def checkExampleWords(inp,dic):
    num     = inp.shape[0]
    for temp in range(0,num):
        if inp[temp] in dic:
            print(dic[inp[temp]],end=" ")
        else:
            print('?',end=" ")            
            
def sampleFromRecipes(recs):
    out     = []
    for recipe in recs:
        l           = max(0,len(recipe) - stripLen)
        startInd    = int((np.random.rand()<probStart)*l*np.random.rand())
        if np.random.rand() < probStart:            
            startInd    = int(min(max(len(recipe)-stripLen,0),recipe.find("name")))
        out.append(recipe[startInd:startInd+stripLen])
    return out

def recipesToMatrix(vocab,word_indices,recipes):
    out     = []
    
    for recipe in recipes:
        example = []
        wt      = wordTokenize(recipe)
        for w in wt:
            if w in word_indices:
                example.append(word_indices[w])
            else:
                example.append(0)
                
        out.append(np.array(example))
        print(example)
        stop=raw_input("")

    return out

def recsToXY(recs):
    X     = np.zeros((len(recs),maxlen))
    y     = np.zeros((len(recs),len(vocab)+1))
    ml     = np.max([r.shape[0] for r in recs])
    toGuess=0
    while toGuess == 0:
        toGuess= np.floor(np.random.rand()*ml)
    for n,rec in enumerate(recs):
        tmpX     = rec[:toGuess]
        tmpy     = np.zeros((len(vocab)+1))
        tmpy[rec[toGuess] ] = 1
        X[n,-tmpX.shape[0]:] = tmpX
        y[n]     = tmpy
    return X, y

def loadThatModel(folder):
    with open(folder+".json",'rb') as f:
        json_string     = f.read()
    model = model_from_json(json_string)
    model.load_weights(folder+".h5")
    return model


def wordTokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    #ends    = re.compile("[.!,- %\n]")
    punc    = re.compile("[()]")
    sent    = sent.replace(".",' . ').replace(',',' , ').replace(':'," : ")
    sent    = re.sub(punc,'',sent)
    patt    = re.compile("[\n ]?")
    nums    = re.compile("\d+")
    sent    = re.sub(nums,"num",sent)
    if sent[-1] in (' ','\n','.',','):
        sent += "garbage"
    r       = [x for x in re.split(patt,sent) if x != '']
    return r

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
toks    = wordTokenize(text)
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
probStart= 0.25

#define model
print(sorted(word_indices.values())[:10])


#wordModel = Sequential()
#wordModel.add(Embedding(vocSize+1, 512,mask_zero=True, input_length=maxlen,input_shape=(maxlen,)))
#wordModel.add(LSTM(512, return_sequences=True,stateful=True))    
#wordModel.add(Dropout(.2))
#wordModel.add(LSTM(512, return_sequences=False,stateful=True))
#
##wordModel.add(RepeatVector(maxlen))
#
#wordModel.add(Dense(len(vocab)+1))
#wordModel.add(Activation('softmax'))
#
#wordModel.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#create a batch of recipes
ind         = 0
epochLoss   = []
recMatrix   = recipesToMatrix(vocab,word_indices,recipes)
while True: 
    
    recs     = recMatrix[ind:ind+batchSize]
    X,y     = recsToXY(recs)    
    print(X.shape)
    print(y.shape)
    stop=raw_input("")
    print(X)
    stop=raw_input("")
    print(y)
    
    ind     = ind+batchSize
    


    if (ind / batchSize) % 50 == 0:
        print()
        jsonstring  = model.to_json()
        with open("../models/recipeRNNMergedStateDeep.json",'wb') as f:
            f.write(jsonstring)
        model.save_weights("../models/recipeRNNMergedStateDeep.h5",overwrite=True)