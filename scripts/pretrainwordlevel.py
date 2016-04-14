from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense,RepeatVector,Merge
from keras.layers.recurrent import GRU,LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import re
import operator
import helper
from collections import Counter
import sys
from keras.models import model_from_json

"""
script to train a word-level LSTM or GRU to predict the next word based on 
the previous hist words
"""


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

    return out

def recsToXY(recs):
    numExamples     = 0
    for r in recs:
        numExamples += (r.shape[0]-maxWord -1)/step
    xlist       = []
    ylist       = []
    y           = np.zeros((numExamples,vocSize))
    addedInd    = 0
    for n,rec in enumerate(recs):
        for m in range(maxWord,rec.shape[0]-1,step):
            
            tmpX     = rec[m-maxWord:m]
            tmpy     = np.zeros((vocSize))
            
            tmpy[rec[m+1] ]             = 1
            ylist.append(tmpy)
            xlist.append(tmpX)
            addedInd+=1

    X   = np.reshape(xlist,(len(xlist),maxWord))
    y   = np.reshape(ylist,(len(ylist),vocSize))
    return X, y



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
batchSize   = 64


#read in the text file
path    = "../allrecipes.txt"
text    = open(path).read().lower()
recipes = [r+"$$$$" for r in text.split("$$$$") if (len(r) > 100 and len(r)<600)]
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

ls     = [len(wordTokenize(rec)) for rec in recipes]
print(np.max(ls))
print(np.min(ls))

counts  = Counter(toks)
vocab   = [x[0] for x in counts.most_common() if x[1] > word_thr]
vocab.insert(0,"oov")
vocSize = len(vocab)
print('corpus length (characters):', len(text))
print('corpus length (tokens)', )
print('vocab size:', vocSize)

word_indices = dict((c, i) for i, c in enumerate(vocab))
indices_word = dict((i, c) for i, c in enumerate(vocab))

maxWord  = 30
step     = 1
#probStart= 0.25
if len(sys.argv) > 1:
    wordModel   = helper.loadThatModel(sys.argv[1])

else:
    wordModel = Sequential()
    wordModel.add(Embedding(vocSize, 256, input_length=maxWord))
    wordModel.add(Dropout(0.2))
    wordModel.add(LSTM(512, return_sequences=True))    
    wordModel.add(Dropout(.2))
    wordModel.add(LSTM(512, return_sequences=False))
    #wordModel.add(RepeatVector(maxlen))
    
    wordModel.add(Dense(vocSize))
    wordModel.add(Activation('softmax'))
    
    wordModel.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#create a batch of recipes
ind         = 0
epochLoss   = []
recMatrix   = recipesToMatrix(vocab,word_indices,recipes)
print(len(recMatrix))
while True: 
    
    if ind> len(recMatrix):
        ind     = 0
        np.random.shuffle(recMatrix)
    
    recs    = recMatrix[ind:ind+batchSize]
    X,y     = recsToXY(recs)    
    
    try:
        wordModel.fit(X,y,nb_epoch=1)    
    except Exception as e:
        print(e)
    
    ind     = ind+batchSize
    


    if (ind / batchSize) % 50 == 0:
        print()
        jsonstring  = wordModel.to_json()
        with open("../models/pretrainedWord.json",'wb') as f:
            f.write(jsonstring)
        wordModel.save_weights("../models/pretrainedWord.h5",overwrite=True)