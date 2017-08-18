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
import gensim
import random

import re

NAMEREGEX = "name:\n(.+?)\n"

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


class Preprocessor:
    
    def __init__(self):
        self.recipes = []


    def recipesFromFile(self, fname):
        """
        read in recipes from .txt, create Recipes from them
        """
        with open(fname, 'r') as f:
            recipeStrings = f.read().split("$$$$")

        for recipeString in recipeStrings:
            recipe = Recipe(recipeString)
            self.recipes.append(recipe)


    def sendToServer(message):
        pass



def parseDocsJar(docs, tempPath, jarLoc = "../java/FlatFileParser.jar",
                 docDelim="ENDDOC", gigsMem=2, njobs=8):
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

    `jarLoc`: str - path to the .jar (should be in U0007)

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
                token.append(word)
                token.append(pos)
                token.append(ner)
                tokens.append(token)
    return documents
