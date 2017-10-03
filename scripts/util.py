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

NAMEREGEX = "name:\n(.+?)\n"
DOCDELIM = "ENDDOC"
# using these constants saves us from the overhead of storing in dictionaries
WORD = 0
POS = 1
NER = 2


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
    
    def __init__(self):
        self.recipes = {}


    def recipesFromFile(self, fname="../data/allrecipes.txt"):
        """
        read in recipes from .txt, create Recipes from them
        """
        with open(fname, 'r') as f:
            recipeStrings = [i for i in f.read().split("$$$$") if i != '']

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


    def buildDict(self, topn=2500):
        self.vocab = {}
        self.NERvocab = {}
        self.POSvocab = {}
        
        for ID, recipe in self.recipes.items():
            for wordTuple in recipe.parsedName:
                self.addToDict(wordTuple)
            for ingredientSet in recipe.parsedIngredients:
                for wordTuple in ingredientSet:
                    self.addToDict(wordTuple)
            for sentence in recipe.parsedDirections:
                for wordTuple in sentence:
                    self.addToDict(wordTuple)
        s = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        p = sorted(self.POSvocab.items(), key=lambda x: x[1], reverse=True)
        nv = sorted(self.NERvocab.items(), key=lambda x: x[1], reverse=True)
        m = np.min([len(s),topn])
        self.vocab = {s[n][0]: n+2 for n in range(0,m)}
        self.vocabReverse = {v: k for k,v in self.vocab.items()}
        self.NERvocab = {p[n][0]: n for n in range(0,len(p))}
        self.NERreverse = {v: k for k,v in self.NERvocab.items()}
        self.POSvocab = {nv[n][0]: n for n in range(0,len(nv))}
        self.POSreverse = {v: k for k,v in self.POSvocab.items()}
        
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
                    mat[n, self.vocabReverse[word]] = 1
                except KeyError:
                    pass
        return mat


def defineModel():
    pass


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


def recipeToMatrices(recipe, prep, stride=20):
    """
    Function to generate matrices from a Preprocessor object full of recipes
    """
    