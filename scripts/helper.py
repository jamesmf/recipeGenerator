from __future__ import print_function
import numpy as np
import re
from collections import Counter
import sys
from keras.models import model_from_json
import cPickle
from os.path import isfile

"""
utility functions for text prediction/generation
"""

class ModelHelper:
    
    def __init__(self,corpus):
        if type(corpus) == list:
            corpus  = ' '.join(corpus)
        self.chars          = list(set(corpus))
        self.numChars       = len(self.chars)
        self.char_indices   = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char   = dict((i, c) for i, c in enumerate(self.chars))
        self.wordTokenizer  = wordTokenize
        self.flex           = {}
        self.tf_threshold   = 2


    def fitVocab(self,corpus,tf_threshold=2):
        self.tf_threshold   = tf_threshold
        toks                = self.wordTokenizer(corpus)
        counts              = Counter(toks)
        self.vocab          = [x[0] for x in counts.most_common() if x[1] > tf_threshold]
        self.vocSize        = len(self.vocab)
        self.word_indices   = dict((c, i+1) for i, c in enumerate(self.vocab))
        
        
    def add(self,name,value):
        """for storing arbitrary data/functions (like a whole LDA model)"""
        self.flex[name]    = value

    def get(self,name):
        return self.flex[name]
        
    def save(self,location,overwrite=False):
        print("saving")
        location += ".pickle"
        a   = "!!!"
        if not overwrite:
            if isfile(location):
                while a not in ('y','n'):
                    a   = raw_input("That ModelHelper Already Exists - Overwrite? y/n")
                    a   = a.strip().lower()
            else:
                a = 'y'
        else:
            a   = 'y'
            
        print(a)
        if a == 'y':
            print("saving as:",location)
            with open(location,'wb') as f:
                cp  = cPickle.Pickler(f)
                cp.dump(self)
                

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / (np.sum(np.exp(a))+0.0001)
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
    if type(dic) == list:
        flag    = True
    else:
        flag    = False
    for temp in range(0,num):
        if inp[temp] in dic:
            print(dic[inp[temp]],end=" ")
        elif flag:
            index   = max(0,int(inp[temp])-1)
            print(dic[index],end=" ")
        else:
            print('?',end=" ")            
            
            
            
def loadThatModel(folder):
    with open(folder+".json",'rb') as f:
        json_string     = f.read()
    model = model_from_json(json_string)
    model.load_weights(folder+".h5")
    return model
    
def loadHelper(folder):
    with open(folder+".pickle",'rb') as f:
        mh  = cPickle.load(f)
    return mh


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
    if sent[-1] in (' ','\n','.',',',':'):
        sent += "garbage"
    r       = [x for x in re.split(patt,sent) if x != '']
    return r[:-1]
    
def getNumberIngredients(recipes):
    out     = []
    for recipe in recipes:
        ingSec  = recipe[recipe.find("ingredients:"):recipe.find("directions:")]
        count   = ingSec.count("\n")
        count   -=3
        out.append(count)
    return out
    
def shuffleIngredients(recipes):
    out     = []
    for recipe in recipes:
        ingSec  = recipe[recipe.find("ingredients:"):recipe.find("directions:")]
        sp      = [ing for ing in ingSec.split("\n") if ing not in ('','ingredients:')]
        np.random.shuffle(sp)
        jo      = "\n".join(sp)
        ingSec2 = "ingredients:\n\n"+jo+"\n\n"
        rec2    = recipe.replace(ingSec,ingSec2)
        out.append(rec2)
    return out
        
def getNames(recipes):
    out     = []
    for recipe in recipes:
        name    = recipe[:recipe.find("ingredients:")-1]
        out.append(name)
    return out

def encodeWord(wordTokens,word_indices,maxWord):
        out     = np.zeros((maxWord,))
        for i,w in enumerate(wordTokens):
            if w in word_indices:
                out[i] = word_indices[w]
        return out

def encodeChar(charSnippet,char_indices):
    """takes in a string, one-hot encodes it based on char_indices"""
    
    lci     = len(char_indices)
    out     = []
    for char in charSnippet:
        zs  =   np.zeros((lci,))
        zs[char_indices[char]] = 1
        out.append(zs)
    out     = np.reshape(out,(len(out),lci))
    return out
    
def getCharAndWordNoState(recipeBatch,conVecBatch,maxlen,maxWord,char_indices,word_indices,step=3,predict=False):
    lci     = len(char_indices)
    
    #initialize lists (to become arrays) for characters, words, and context vectors
    Xcontext    = []
    Xcharacter  = []
    Xword       = []
    ys          = []

    #loop over recipes
    for rnum, recipe in enumerate(recipeBatch):
        if predict:
            recipe = recipe+" "
        conVec  = conVecBatch[rnum] #get the context vector for the whole recipe
        
        #step through the document, taking snippets
        startingPoint   = 0
        if predict:
            startingPoint   = len(recipe)-maxlen-1 
        for exnum in range(0,len(recipe)-maxlen,step):
            #define a maxlen-length character snippet
            charSnippet     = recipe[exnum: exnum+maxlen+1]
            
            #convert the text from the beginning of the recipe to the index into words
            wordSnippet     = recipe[0: exnum+maxlen]
            wordTokens      = wordTokenize(wordSnippet)
            wordTokens      = wordTokens[-maxWord:]  # keep only the last maxWord tokens
            
            #text snippet to one-hot encoded characters
            charEncoded     = encodeChar(charSnippet,char_indices)
            #target = last index and [:-1] is input
            xchar           = charEncoded[:-1]
            y               = charEncoded[-1]
            
            #turn word tokens into their indices
            wordEncoded     = encodeWord(wordTokens,word_indices,maxWord)

            #append vectors to their respective arrays
            Xcontext.append(conVec)
            Xcharacter.append(xchar)
            Xword.append(wordEncoded)
            ys.append(y)
    
    #reshape the arrays    
    Xcontext    = np.reshape(Xcontext,(len(Xcontext),len(conVec)))
    Xcharacter  = np.reshape(Xcharacter,(len(Xcharacter),xchar.shape[0],xchar.shape[1]))
    Xword       = np.reshape(Xword, (len(Xword),wordEncoded.shape[0]))
    ys           = np.reshape(ys, (len(ys),lci))

    return Xcharacter, Xword, Xcontext, ys
    
    
    
def getTrainOfThought(text):
    pass    