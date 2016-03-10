from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense,RepeatVector,Merge
from keras.layers.recurrent import GRU
from keras.datasets.data_utils import get_file
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
    a = a.astype(float) -0.00001
    #print(np.sum(a[:-1]))
    return np.argmax(np.random.multinomial(1, a, 1))

def checkExample(inp,dic):
    if inp.shape[0] == len(dic):
        inp     = np.reshape(inp,(1,inp.shape[0]))
    print(inp.shape)
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
        out.append(recipe[startInd:startInd+stripLen])
    return out

def recipesToChars(chars,char_indices,recipes):
    Xchar   = []
    Ychar   = []
    for recipe in recipes:
        char_snips  = []
        #next_chars  = []
        #print(len(recipe), maxlen, step)
        #print(range(0, len(recipe) - maxlen-1, step))
        for i in range(0, len(recipe) - maxlen+1,step):
            #print(i)
            char_snips.append(recipe[i: i + maxlen])
            #next_chars.append(recipe[i + maxlen])

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
            
        Xchar.append(xchar)
        Ychar.append(ychar)
    
    XcharOut   = np.zeros((len(Ychar[0])*batchSize,maxlen,len(char_indices)))
    
    ind     = 0
    added   = 0
    no      = 0
    for i in range(0,len(Ychar[0])):
        for j in range(0,batchSize):
            if (j < len(Xchar)) and (i < len(Xchar[j])):
                XcharOut[ind,:,:] = Xchar[j][i,:]
                added+=1
            else:
                no+=1
            ind+=1

    return XcharOut


def recipesToWords(vocab,word_indices,recipes,recs):
    Xword   = []
    for recnumber, recipe in enumerate(recipes):
        ind1        = recipe.find(recs[recnumber])
        recipe      = recipe[0:ind1+len(recs[recnumber])]
        word_snips  = []
        #next_chars  = []
        for i in range(0, len(recipe) - maxlen+1, step):
            
            word_snips.append(wordTokenize(recipe[: i + maxlen])[:-1])
            #next_chars.append(recipe[i + maxlen])

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

maxlen      = 20
step        = 1
batchSize   = 16


model   = loadThatModel("../models/recipeRNNMergedState")


name        = "cheesy enchiladas\n"
recipes     = ["name:\n"+name for dummy in range(0,batchSize)]
div         = np.linspace(0.2,1.,batchSize)
startLen    = len(recipes[0])

#create a batch of recipes
ind         = 0
epochLoss   = []
for cInd in range(400): 
    if (cInd+maxlen) < startLen:
        recs    = [r[cInd:cInd+maxlen] for r in recipes]
    else:
        recs  = [r[-maxlen:] for r in recipes]
    

    Xchar       = recipesToChars(chars,char_indices,recs)
    Xword       = recipesToWords(vocab,word_indices,recipes,recs)
    Xword       = Xword[-batchSize:]
    #xcharbat    = Xchar[i*batchSize:(i+1)*batchSize,:,:]
    #xwordbat    = Xword[i*batchSize:(i+1)*batchSize,:] 
    #print(Xchar.shape, Xword.shape)

    preds       = model.predict_on_batch([Xchar,Xword])[0]
   
    for d,pred in enumerate(preds):
        #print(d,pred)
        next_index  = sample(pred, div[d])
        next_char   = indices_char[next_index]

        if recipes[d][-1]  != "$":
            if (cInd+maxlen) >= startLen:
                recipes[d] += next_char
    
    dropIt  = [(r[-1] == '$')*1. for r in recipes]
    if int(np.sum(dropIt)) == batchSize:
        break
    
for d,rec in enumerate(recipes):
    print("with diversity:",div[d],"\n\n",rec,'\n\n')