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

def processRecipes(vocab,word_indices,chars,char_indices,recipes,maxlen,step):
    Xchar   = []
    Xword   = []
    Ychar   = []
    for recipe in recipes:
        char_snips  = []
        word_snips  = []
        next_chars  = []
        for i in range(0, len(recipe) - maxlen, step):
            char_snips.append(recipe[i: i + maxlen])
            next_chars.append(recipe[i + maxlen])
            start   = 0
            words   = wordTokenize(recipe[start:i+maxlen])[-maxlen:-1]
            word_snips.append(words)


        #iterate over snippets within one recipe
        xchar   = np.zeros((len(char_snips),maxlen,len(chars)))
        xword   = np.zeros((len(char_snips),maxlen))
        ychar   = np.zeros((len(char_snips),len(chars)))
        for i, charsnip in enumerate(char_snips):
            #get the word history at charsnip i            
            wordsnip= word_snips[i]
            #turn each snippet into a one-hot encoded array of examples x time x output
                    
            #for each timestep in the snippet
            for t, char in enumerate(charsnip):

                #onehot character vector
                x_i     = np.zeros(len(char_indices))
                x_i[char_indices[char]] = 1
                
                #onehot word vector
                x_iw    = 0 
                if len(wordsnip) > t:
                    #print(wordsnip[t])
                    if wordsnip[t] in word_indices:
                        #print("did it!")
                        x_iw = word_indices[word_snips[i][t]]
                    else:
                        #print(wordsnip[t])
                        pass

                xchar[i,t,:]    = x_i
                xword[i,t]    = x_iw
            y_i     = np.zeros(len(chars))
            y_i[char_indices[next_chars[i]]] = 1
            
            
            ychar[i,:] = y_i
            
        Xchar.append(xchar)
        Xword.append(xword)
        Ychar.append(ychar)

        

    
    XcharOut   = np.zeros((len(Ychar[0])*batchSize,maxlen,len(char_indices)))
    XwordOut   = np.zeros((len(Ychar[0])*batchSize,maxlen))
    YcharOut   = np.zeros((len(Ychar[0])*batchSize,len(char_indices)))
    
    ind     = 0
    added   = 0
    no      = 0
    for i in range(0,len(Ychar[0])):
        for j in range(0,batchSize):

            if (j < len(Xchar)) and (i < len(Xchar[j])):
                XcharOut[ind,:,:] = Xchar[j][i,:]
                XwordOut[ind,:] = Xword[j][i]
                YcharOut[ind,:]   = Ychar[j][i]
                added+=1
            else:
                no+=1
            ind+=1

    return XcharOut,XwordOut,YcharOut

def wordTokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    punc    = re.compile("[()]")
    sent    = sent.replace(".",' . ').replace(',',' , ').replace(':'," : ")
    sent    = re.sub(punc,'',sent)
    patt    = re.compile("[\n ]?")
    nums    = re.compile("\d+")
    sent    = re.sub(nums,"num",sent)
    r       = [x for x in re.split(patt,sent) if x != '']
    return r

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
maxlen   = 20
step     = 3



#read in the text file
path    = "allrecipes.txt"
text    = open(path).read().lower()
recipes = [r+"$$$$" for r in text.split("$$$$")]
recipes = sorted(recipes,key=lambda x: len(x),reverse=True)
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






#can't load model with load_from_json
charModel = Sequential()
charModel.add(GRU(128, return_sequences=True,input_shape=(maxlen, len(chars))))
charModel.add(TimeDistributedDense(128))

wordModel = Sequential()
wordModel.add(Embedding(vocSize+1, 128, input_length=maxlen))
wordModel.add(GRU(128, return_sequences=False,))
wordModel.add(RepeatVector(maxlen))

model = Sequential()
model.add(Merge([charModel, wordModel], mode='concat', concat_axis=-1))
model.add(GRU(256, return_sequences=False))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.load_weights("recipeRNNMerged.h5")



# output generated text after each iteration
for iteration in range(1, 10):
    print()
    print('-' * 50)
    print('Iteration', iteration)


    print()

    generated = ''
    #sentence = text[start_index: start_index + maxlen]
    name      = "craigs crazy tacos"
    recipes   = ["name:\n"+name for i in range(0,batchSize)]
    #generated += sentence
    #print('----- Generating with seed: "' + sentence + '"')
    #sys.stdout.write(generated)
    div     = np.linspace(.2,1.5,batchSize)
    for i in range(400):
        xc,xw,y     = processRecipes(vocab,word_indices,chars,char_indices,recipes,maxlen,step)            
        xc          = xc[-batchSize:]
        xw          = xw[-batchSize:]        
        preds = model.predict([xc,xw], verbose=0)
        
        for num,diversity in enumerate(div):
            next_index = sample(preds[num], diversity)
            next_char = indices_char[next_index]
    
            recipes[num] += next_char
            #sentence = sentence[1:] + next_char
    
    for rec in recipes:
        print(rec)
    

    
