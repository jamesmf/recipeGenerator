# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 23:04:50 2017

@author: jmf
"""
import os
import sys
import re
import time
import subprocess
import numpy as np
#import gensim
import random
import re
from scipy import sparse
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Embedding, Conv1D, Dense, MaxPooling1D
from keras.layers import GlobalMaxPooling1D, Concatenate, BatchNormalization
from keras.layers import Activation, Multiply
from keras.models import Model
import keras.optimizers as opt

NAMEREGEX = "name:\n(.+?)\n"
DOCDELIM = "ENDDOC"
# using these constants saves us from the overhead of storing in dictionaries
WORD = 0
NER = 2
POS = 1




class Recipe:
    
    def __init__(self, inp):
        self.ingredientStrings = []
        self.buildFromText(inp)


    def buildFromText(self, txt):
        """
        parse a recipe into its parts
        """
        ind = txt.find("name:")+6
        txt = txt[ind:]
        self.name = txt[:txt.find("\n\n")]
        ingInd = txt.find("ingredients:\n\n")+14
        txt = txt[ingInd:]
        dirInd = txt.find("directions:\n")
        ingSec = txt[:dirInd]
        for line in ingSec.split("\n"):
            if len(line) > 1:
                self.ingredientStrings.append(line.strip())
        self.directionsString = txt[dirInd+12:].strip()


    def toParsable(self):
        txt = "RECIPE_ID "+str(self.recipeID)+'\n'+DOCDELIM+'\n'
        txt += self.name+'\n'+DOCDELIM+'\n'
        txt += ('\n'+DOCDELIM+'\n').join(self.ingredientStrings)
        txt += '\n'+DOCDELIM+'\nENDINGREDIENTS\n'+DOCDELIM+'\n'+self.directionsString
        txt += '\n'+DOCDELIM+'\nENDRECIPE\n'+DOCDELIM+'\n'
        return txt


    def toTokenList(self):
        tokens = [("name","NN","O"),(':',':','O'),('\n',':','MISC')]
        for tup in self.parsedName:
            tokens.append(tup)
            tokens.append((' ',':','O'))
            if tup[WORD]  in ('.','!','?','@',','):
                t = tokens.pop(-3)
        tokens[-1] = ('\n',':','MISC')
        tokens.append(('\n',':','MISC'))
        np.random.shuffle(self.parsedIngredients)
        for line in self.parsedIngredients:
            for tup in line:
                tokens.append(tup)
                tokens.append((' ',':','O'))
                if tup[WORD]  in ('.','!','?','@',','):
                    t = tokens.pop(-3)
            tokens[-1] = ('\n',':','MISC')
        tokens.append(('\n',':','MISC'))
        tokens.append(("directions",'NN','O'))
        tokens.append((':',':','O'))
        for line in self.parsedDirections:
            for tup in line:
                tokens.append(tup)
                tokens.append((' ',':','O'))
                if tup[WORD]  in ('.','!','?','@',','):
                    t = tokens.pop(-3)
        tokens[-1] = ("$$$$",'CD','MISC')
        return tokens
        

class Preprocessor:
    
    def __init__(self, maxWord, maxNgram):
        self.recipes = {}
        self.maxWord = maxWord
        self.maxNgram = maxNgram


    def recipesFromFile(self, fname="../data/allrecipes.txt"):
        """
        read in recipes from .txt, create Recipes from them
        """
        with open(fname, 'r') as f:
            fullDoc = f.read()
            recipeStrings = [i for i in fullDoc.split("$$$$") if i != '']
            self.buildCharNgramDict(fullDoc)

        recipeInd = 0
        for recipeString in recipeStrings:
            recipe = Recipe(recipeString)
            recipe.recipeID = recipeInd
            self.recipes[recipe.recipeID] = recipe
            recipeInd += 1

        self.docs = self.sendToStanfordParser()
        self.readParsedDocs()


    def sendToStanfordParser(self):
        docs = []
        tempPath = "../data/"
        for name, recipe in self.recipes.items():
            docs.append(recipe.toParsable())
        docs = parseDocsJar(docs,tempPath)
        return docs
        

    def readParsedDocs(self):
        location = "ID"
        currentRecipe = None
        for doc in self.docs:
            if location == "ID":
                recipeID = int(doc[0][1][WORD])
                location = "name"
                currentRecipe = self.recipes[recipeID]
            elif location == "name":
                currentRecipe.parsedName = doc[0]
                currentRecipe.parsedIngredients = []
                location = "ingredients"
            elif location == "ingredients":
                if doc[0][0][WORD] == "ENDINGREDIENTS":
                    location = "directions"
                else:
                    reconstructedIngredient = []
                    for sentence in doc:
                        for word in sentence:
                            reconstructedIngredient.append(word)
                    currentRecipe.parsedIngredients.append(reconstructedIngredient)
            elif location == "directions":
                if doc[0][0][WORD] == "ENDRECIPE":
                    location = "ID"
                else:
                    currentRecipe.parsedDirections = doc


    def buildCharNgramDict(self, fullDoc, topn=50000):
        """
        builds a dictionary of ngrams of size 3 (may make this a range)
        """
        self.ngramVocab = {}
        for ngramSize in range(3,4):
            z = zip(*[fullDoc[i:] for i in range(0,ngramSize)])
            for ng in z:
                ngram = ''.join(ng)
                try:
                    self.ngramVocab[ngram] += 1
                except KeyError:
                    self.ngramVocab[ngram] = 1

        s = sorted(self.ngramVocab.items(), key=lambda x: x[1], reverse=True)
        m = np.min([len(s), topn])
        self.ngramVocab = {s[n][0]: n+2 for n in range(0,m)}
        self.ngramVocabReverse = {v: k for k,v in self.ngramVocab.items()}


    def buildDict(self, topn=10000):
        self.vocab = {}
        self.NERvocab = {}
        self.POSvocab = {}
        
        for ID, recipe in self.recipes.items():
            t = recipe.toTokenList()
            for token in t:
                self.addToDict(token)
            
        s = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        p = sorted(self.POSvocab.items(), key=lambda x: x[1], reverse=True)
        nv = sorted(self.NERvocab.items(), key=lambda x: x[1], reverse=True)
        m = np.min([len(s), topn])
        self.vocab = {s[n][0]: n+2 for n in range(0,m)}
        self.vocab["OOV"] = 1
        self.vocab["MASKED"] = 0
        self.vocabReverse = {v: k for k,v in self.vocab.items()}
        self.POSvocab = {p[n][0]: n for n in range(0,len(p))}
        self.POSreverse = {v: k for k,v in self.NERvocab.items()}
        self.NERvocab = {nv[n][0]: n for n in range(0,len(nv))}
        self.NERreverse = {v: k for k,v in self.POSvocab.items()}
        
        self.charDict = {}
        self.charDictReverse = {}
        lowercase = [chr(i) for i in range(97,123)]
        punct = ['[',']',',',' ','?',':',';','+','-','.','!','*','(',')','@',
                 '"',"'",'&','/']
        chars = lowercase + punct + [str(i) for i in range(0,10)]
        for c in chars:
            l = len(self.charDict)+1
            self.charDict[c] = l
            self.charDictReverse[l] = c
        self.charDict[''] = 0
        self.charDictReverse[0] = ''


    def getWordVecs(self, fname="../data/glove.6B.100d.txt"):
        """
        Load up pretrained vectors for words in vocabulary (currently GloVe)
        """
        self.vectors = {}
        with open (fname, 'r') as f:
            for line in f:
                s = line.split(' ')
                word = s[0]
                vector = np.array(s[1:])
                if word in self.vocab:
                    self.vectors[word] = vector
        print(len(self.vectors)*1./len(self.vocab))
        for word in self.vocab.keys():
            if word not in self.vectors:
                self.vectors[word] = np.array([np.random.rand()/50 for i in range(0,100)])


    def addToDict(self, wordTuple):
        word = wordTuple[WORD]
        pos = wordTuple[POS]
        ner = wordTuple[NER]
        if word in self.vocab:
            self.vocab[word] += 1
        else:
            self.vocab[word] = 1
        if ner in self.NERvocab:
            self.NERvocab[ner] += 1
        else:
            self.NERvocab[ner] = 1
        if pos in self.POSvocab:
            self.POSvocab[pos] += 1
        else:
            self.POSvocab[pos] = 1


    def toSparseMatrix(self, recipeInds):
        """
        create a sparse matrix out of the recipes based on an input list of 
        indices/keys of which recipes
        
        used to generate matrices for LDA
        """
        mat = sparse.lil_matrix((len(recipeInds),len(self.vocab)))
        for n, ind in enumerate(recipeInds):
            rec = self.recipes[ind]
            par = rec.toTokenList()
            for tup in par:
                word = tup[0]
                try:
                    mat[n, self.vocab[word]] += 1
                except KeyError:
                    pass
        return mat


def recipeToExample(prep, ind, strideMax=10):
    ldavec = prep.LDAvecs[ind]
    r = prep.recipes[ind]
    exampleNgrams = []
    exampleWords = []
    exampleOutChars = []
    exampleOutPOS = []
        
    tokens = r.toTokenList()
    wordLoc = 2
    firstWord = getIndex(prep, tokens[0][WORD])
    secondWord = getIndex(prep, tokens[1][WORD])
    thirdWord = getIndex(prep, tokens[2][WORD])
    runningString = tokens[0][WORD]+tokens[1][WORD]+tokens[2][WORD]
    words = [firstWord, secondWord, thirdWord]
    stride = np.random.randint(1, strideMax)
#    print(tokens)
#    print(words)
#    print(stride)
    while wordLoc < len(tokens):
        word = tokens[wordLoc][WORD]
#        print(wordLoc, word, stride)
        if stride >= len(word):
            wordLoc += 1
            runningString += word
            words.append(getIndex(prep, word))
            stride -= len(word)
        else:
            ngrams = getNgrams(prep, runningString+word[:stride])
            pastWords = pad_sequences([words], prep.maxWord)[0]
            targetChar = getCharIndex(prep, word[stride])
            target = to_categorical(targetChar, num_classes=len(prep.charDict))
            pos = getPOS(prep, tokens[wordLoc][POS])
            posTarget = to_categorical(pos, num_classes=len(prep.POSvocab))
            exampleOutPOS.append(posTarget)
            exampleOutChars.append(target)
            exampleNgrams.append(ngrams)
            exampleWords.append(pastWords)
#            print("running: ", runningString)
#            print("words: ",words)
#            print("translated: ",[prep.vocabReverse[i] for i in words])
#            print("ngrams: ", ngrams)
#            print("target char:", word[stride])
#            print("POS: ", posTarget, pos)
            stride = np.random.randint(1, strideMax) + stride        

    lda = np.multiply(np.ones((len(exampleOutPOS), ldavec.shape[0])), ldavec)
    exampleNgrams = np.array(exampleNgrams)
    exampleWords = np.array(exampleWords)
    exampleOutPOS = np.array(exampleOutPOS).reshape((len(exampleOutPOS),len(prep.POSvocab)))
    exampleOutChars = np.array(exampleOutChars).reshape((len(exampleOutChars),len(prep.charDict)))
    
    return exampleNgrams,exampleWords, lda, exampleOutChars, exampleOutPOS



def getNgrams(prep, rs):
    rs = rs[-(prep.maxNgram+3):]
    rs = [''.join(x) for x in zip(*[rs[i:] for i in range(0,3)])]
    out = []
    for ngram in rs:
        try:
            out.append(prep.ngramVocab[ngram])
        except KeyError:
            out.append(1)
    return pad_sequences([out], prep.maxNgram)[0]


def getIndex(prep, token):
    w = token
    try:
        return prep.vocab[w]
    except KeyError:
        return 1


def getCharIndex(prep, char):
    try:
        return prep.charDict[char]
    except KeyError:
        return 0


def getPOS(prep, pos):
    try:
        return prep.POSvocab[pos]
    except KeyError:
        return 0


def parseDocsJar(docs, tempPath, jarLoc = "../java/FlatFileParser.jar",
                 docDelim="ENDDOC", gigsMem=1, njobs=4):
    """
    Function to parse documents into sentences, then sentences into tokens.
    The data will be dumped to disk (ASCII-encoded), then read in by the .jar
    which will write out documents, separated by docDelim, and sentences, 
    separated by 'SENT', each consisting of tuples of (lemma,POS,word,NER).
    This output is read in by readInParsedFile() and returned as a dict of docs
    each of which contains sentences, which are lists of token dicts.
    
    input:

    `docs`: list - list of free text documents to be parsed

    `tempPath`: str - path where the temp file should be dumped. Will be deleted

    `jarLoc`: str - path to the .jar 

    `docDelim`: str - document delimiter

    `gigsMem`: int - number of gigs of memory to allow Java

    `njobs`: int - number of processes to spin up at once

    returns:

    `docs`: dict of documents (lists of lists of tokens). Tokens are dicts with
    keys of ('ner', 'pos', 'word', 'lemma')
    """
    # prepare for multiprocess
    processes = set()
    filenames = []

    #split the docs for multiple jobs
    ind = 0
    docsPer = divmod(len(docs),njobs)[0]+1

    # dump sentences to disk for Java process to consume
    for i in range(0, njobs):
        tempdocs = docs[ind:ind+docsPer]
        if len(tempdocs) > 0 :
            name = "\\dump_"+str(random.randint(0,100000000))+".txt"
            with open(tempPath+name,'w') as f:
                for doc in tempdocs:
                    f.write(doc.encode('UTF-8').decode('ASCII','ignore'))
                    f.write("\n"+docDelim+"\n")
    
            call = "java -jar -Xmx"+str(gigsMem)+"g "+'"'+jarLoc+'" "'+tempPath+name+'"'
            print("starting process: " + call)
            processes.add(subprocess.Popen(call,shell=True))
    
            filenames.append(tempPath+name)
            ind += docsPer

    while len(processes) > 0:
        time.sleep(.5)
        processes.difference_update([p for p in processes if p.poll() is not None])

    docs = []
    for filename in filenames:
        tempdocs = readInParsedFile(filename+".out",docDelim)
        docs += tempdocs
        try:
            os.remove(filename)
            os.remove(filename+".out")
        except OSError:
            pass
    return docs
    
    
def readInParsedFile(fn, docDelim):
    """
    Function to work with parseDocsJar to convert .jar output to dict
    """
    documents = []
    doc = []
    tokens = []
    with open(fn,'r') as f:
        for line in f:
            line = line.strip()
            if line == docDelim:
                if len(doc) > 0:
                    documents.append(doc)
                doc = []
            elif line == "SENT":
                if len(tokens) > 0:
                    doc.append(tokens)
                tokens = []
            else:
                token = []
                (word,pos,ner) = line.split('\t')
                if word == '-LRB-':
                    word = '('
                if word == '-RRB-':
                    word = ')'
                token.append(word)
                token.append(pos)
                token.append(ner)
                tokens.append(token)
    return documents


def splitDocs(docs, ratio=0.8):
    numTrain = int(ratio*len(docs))
    return docs[:numTrain], docs[numTrain:]


def recipesToMatrices(prep, inds, stride=15):
    """
    Function to generate matrices from a Preprocessor object full of recipes
    """
    ngramList = []
    wordList = []
    vecsList = []
    charList = []
    posList = []
    relevantLists = [ngramList, wordList, vecsList, charList, posList]
    fullLength = 0
    for n, ind in enumerate(inds):
        result = recipeToExample(prep, ind, strideMax=stride)
        for i in range(0,5):
            for item in result[i]:
                relevantLists[i].append(item)
        fullLength += len(result[0])
    print(fullLength)
    ngramMat = np.zeros((fullLength, ngramList[0].shape[0]))
    wordMat = np.zeros((fullLength, wordList[0].shape[0]))
    vecMat = np.zeros((fullLength, vecsList[0].shape[0]))
    charMat = np.zeros((fullLength, charList[0].shape[0]))
    posMat = np.zeros((fullLength, posList[0].shape[0]))
    
    relevantMatrices = [ngramMat, wordMat, vecMat, charMat, posMat]
    for i in range(0, fullLength):
        for j in range(0, 5):
            relevantMatrices[j][i,:] = relevantLists[j][i]

    return ngramMat, wordMat, vecMat, charMat, posMat


def addBlock(inp, blockNum, dilation, woc):
    bn = str(blockNum)
    conva = Conv1D(32, 3, padding='causal',
                   dilation_rate=dilation, name='conv_'+woc+'_'+bn+'a')(inp)
    conva = BatchNormalization(axis=2, scale=False,
                                name='batchnorm_'+woc+'_'+bn+'a')(conva)
    convb = Conv1D(32, 3, padding='causal',
                   dilation_rate=dilation, name='conv_'+woc+'_'+bn+'b')(inp)
    convb = BatchNormalization(axis=2, scale=False,
                                name='batchnorm_'+woc+'_'+bn+'b')(convb)
    convb = Activation('sigmoid', name='gate_'+woc+'_'+bn+'b')(convb)
    conv = Multiply(name='block_'+woc+'_'+bn)([conva, convb])
    return conv    

def defineModel(prep, numCharLayers=3, numWordLayers=3):
    
    maxNgram = prep.maxNgram 
    maxWord = prep.maxWord
    ngramVocabSize = len(prep.ngramVocab)
    wordVocabSize = len(prep.vocab)
    # character ngram input
    inp1 = Input(shape=(maxNgram,), name='ngram_input')
    cLayer = Embedding(ngramVocabSize, 12, name='ngram_embedding',
                       input_length=maxNgram)(inp1)

    inp2 = Input(shape=(maxWord,), name='word_input')
    wLayer = Embedding(wordVocabSize, 100, name='word_embedding',
                       input_length=maxWord)(inp2)
    
    for i in range(0, numCharLayers):
        cLayer = addBlock(cLayer, i, 2**i, 'ngram')
    ngram = GlobalMaxPooling1D()(cLayer)
    
    for i in range(0, numWordLayers):
        wLayer = addBlock(wLayer, i, 2**i, 'word')
    word = GlobalMaxPooling1D()(wLayer)
        
 
    inp3 = Input(shape=(prep.LDAsize,), name='LDA_vec_input')
    gen = Dense(64, activation='sigmoid', name='gen_mask')(inp3)
    
    merged = Concatenate(name='concat_ngram_word')([ngram,word])
    fc = Dense(64, name='fc', activation='relu')(merged)
    fc = Multiply(name='multiply_fc_by_gen')([gen, fc])
    
    pos = Dense(len(prep.POSvocab), name='POS', activation='sigmoid')(fc)
    
    sf = Dense(len(prep.charDict), name='char_out', activation='sigmoid')(fc)
    
    
    model = Model([inp1, inp2, inp3], [sf, pos])
    rms = opt.RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms,
                  loss_weights=[1., 0.5])
    
    return model