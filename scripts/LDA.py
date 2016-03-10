from __future__ import print_function
import numpy as np
import gensim as gs
import re
import operator
#from collections import Counter
import sys
from sklearn.feature_extraction.text import CountVectorizer
"""
script to train a word-level LSTM or GRU to predict the next word based on 
the previous hist words
"""


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


#read in the text file
path    = "../allrecipes.txt"
text    = open(path).read().lower()
recipes = [r+"$$$$" for r in text.split("$$$$")]
np.random.shuffle(recipes)
print("number of recipes:",len(recipes))


#define the word vocabulary
#word_thr= 2
#toks    = wordTokenize(text)
#counts  = Counter(toks)
#vocab   = [x[0] for x in counts.most_common() if x[1] > word_thr]
#vocSize = len(vocab)


#word_indices = dict((c, i+1) for i, c in enumerate(vocab))
#indices_word = dict((i+1, c) for i, c in enumerate(vocab))

vect = CountVectorizer(min_df=2, ngram_range=(1, 1), max_features=25000)
corpus_vect = vect.fit_transform(recipes)
print(vect.vocabulary_)
id2w     = {}
for k,v in vect.vocabulary_.iteritems():
    id2w[v] = k

print("dictionary complete")
# transform sparse matrix into gensim corpus
corpus    = gs.matutils.Sparse2Corpus(corpus_vect, documents_columns=False)
print("to corpus")
lda     = gs.models.ldamodel.LdaModel(corpus=corpus, id2word=id2w, num_topics=25, update_every=0, chunksize=300, passes=20)
# I instead would like something like this line below
# lsi = gensim.models.LsiModel(corpus_vect_gensim, id2word=vect.vocabulary_, num_topics=2)
print(lda.print_topics(10))
for n,vec in enumerate(corpus):
    print(lda[vec])
    print(recipes[n][:100])
#['0.622*"21" + 0.359*"31" + 0.256
