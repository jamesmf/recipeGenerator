'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
import numpy as np
import sys
from keras.models import model_from_json
import re
from collections import Counter
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense,RepeatVector,Merge
from keras.layers.recurrent import GRU

def checkExample(inp,dic):
    if inp.shape[0] == len(dic):
        inp     = np.reshape(inp,(1,inp.shape[0]))
    #print(inp.shape)
    num     = inp.shape[0]
    for temp in range(0,num):
        #print(i    np[temp])
        #print(np.sum(inp[temp]))
        if np.sum(inp[temp]) > 0:
            #print(list(inp[temp]))
            #print(list(inp[temp]).index(1))
            print(dic[list(inp[temp]).index(1)],end="")
    print()

def processRecipes(chars,char_indices,recipes,maxlen,step):

    XcharOut= np.zeros((batchSize,maxlen,len(char_indices)))
    for rnum, recipe in enumerate(recipes):
        charsnip    = recipe
        xchar   = np.zeros((1,maxlen,len(chars)))
        #for each timestep in the snippet
        for t, char in enumerate(charsnip):

            #onehot character vector
            x_i     = np.zeros(len(char_indices))
            x_i[char_indices[char]] = 1

            xchar[0,t,:]    = x_i
        XcharOut[rnum] = xchar    
    return XcharOut

def loadThatModel(folder):
    with open(folder+".json",'rb') as f:
        json_string     = f.read()
    model = model_from_json(json_string)
    model.load_weights(folder+".h5")
    return model


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))



#in order to use stateful RNNs, we process in batches
batchSize   = 16
maxlen      = 20
step        = 3
folder      = sys.argv[1]


#read in the text file
path    = "../allrecipes.txt"
text    = open(path).read().lower()
recipes = [r+"$$$$" for r in text.split("$$$$")]
recipes = sorted(recipes,key=lambda x: len(x),reverse=True)
print("number of recipes:",len(recipes))

#define the character vocabulary
chars = list(set(text))
print(chars)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#define the word vocabulary
word_thr= 2

model   = loadThatModel(folder)
model.reset_states()


# output generated text after each iteration
generated = ''
#sentence = text[start_index: start_index + maxlen]
name      = "vegetarian tacos\n"
recipes   = ["name:\n"+name for i in range(0,batchSize)]
#generated += sentence
#print('----- Generating with seed: "' + sentence + '"')
#sys.stdout.write(generated)
div     = np.linspace(0.2,1.2,batchSize)
startLen= len(recipes[0])
for i in range(400):
    if (i+maxlen) < startLen:
        recs    = [r[i:i+maxlen] for r in recipes]
    else:
        recs  = [r[-maxlen:] for r in recipes]
    xc  = processRecipes(chars,char_indices,recs,maxlen,step)    
    #print(xc.shape)        
#    print(recs[0])
#    print(checkExample(xc[0],chars))
#    stop=raw_input("")
#    if i == 0:
#        xc  = xc[:batchSize]
#    else:
#        xc    = xc[-batchSize:]
    preds = model.predict(xc, verbose=0)
    
    for d,pred in enumerate(preds):
        next_index = sample(pred, div[d])
        next_char = indices_char[next_index]
        #print(next_char)
        #stop=raw_input("")
        if recipes[d][-1]  != "$":
            if (i+maxlen) >= startLen:
                recipes[d] += next_char
        #sentence = sentence[1:] + next_char
    
for d,rec in enumerate(recipes):
    print("with diversity:",div[d],"\n",rec)
    
    
    
