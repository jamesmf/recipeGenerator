from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import GRU
from keras.datasets.data_utils import get_file
import numpy as np
import re
import random
import sys
#from keras.models import model_from_json

"""
script to train a word-level LSTM or GRU to predict the next word based on 
the previous hist words
"""


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]

path    = "allrecipes.txt"
text    = open(path).read().lower()
toks    = tokenize(text)
tokSet  = set(toks)
vocSize = len(tokSet)
print('corpus length:', len(text))
print('unique tokens:', len(tokSet))

word_indices = dict((c, i) for i, c in enumerate(tokSet))
indices_word = dict((i, c) for i, c in enumerate(tokSet))

maxlen  = 30
step    = 3
sentences   = []
next_words  = []
