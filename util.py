# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:41:36 2018

@author: rishabh
"""

import numpy as np
import string
import os
import sys
import operator
from nltk import pos_tag, word_tokenize
from datetime import datetime

def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

def all_parity_pairs(nbit):
    # total number of samples (Ntotal) will be a multiple of 100
    # why did I make it this way? I don't remember.
    N = 2**nbit
    remainder = 100 - (N % 100)
    Ntotal = N + remainder
    X = np.zeros((Ntotal, nbit))
    Y = np.zeros(Ntotal)
    for ii in range(Ntotal):
        i = ii % N
        # now generate the ith sample
        for j in range(nbit):
            if i % (2**(j+1)) != 0:
                i -= 2**j
                X[ii,j] = 1
        Y[ii] = X[ii].sum() % 2
    return X, Y

def all_parity_pairs_with_sequence_labels(nbit):
    X, Y = all_parity_pairs(nbit)
    N, t = X.shape

    # we want every time step to have a label
    Y_t = np.zeros(X.shape, dtype=np.int32)
    for n in range(N):
        ones_count = 0
        for i in range(t):
            if X[n,i] == 1:
                ones_count += 1
            if ones_count % 2 == 1:
                Y_t[n,i] = 1

    X = X.reshape(N, t, 1).astype(np.float32)
    return X, Y_t
    
def remove_punctuation(s):
  return s.translate(str.maketrans('','',string.punctuation))
  
def get_robert_frost():
    word2idx={'START':0,'END':1}
    current_idx=2
    sentences=[]
    for line in open('E:/RS/ML/Machine learning tuts/Target/Part4(NLP)/14)[FreeTutorials.Us] deep-learning-recurrent-neural-networks-in-python/Code/robert_frost.txt'):
        line=line.strip()
        if line:
            tokens=remove_punctuation(line.lower()).split()
            sentence=[]
            for t in tokens:
                if t not in word2idx:
                    word2idx[t]=current_idx
                    current_idx+=1
                idx=word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    return sentences,word2idx
''' 
word2idx={'START':0,'END':1}
current_idx=2
sentences=[]
for line in open('E:/RS/ML/Machine learning tuts/Target/Part4(NLP)/14)[FreeTutorials.Us] deep-learning-recurrent-neural-networks-in-python/Code/robert_frost.txt'):
    line=line.strip()
    if line:
        tokens=remove_punctuation(line.lower()).split()
        sentence=[]
        for t in tokens:
            if t not in word2idx:
                word2idx[t]=current_idx
                current_idx+=1
            idx=word2idx[t]
            sentence.append(idx)
        sentences.append(sentence)
V=len(word2idx)
D=30        
We=init_weight(V,D)
sentences[0].shape
sentences
Wout=We[sentences]
Wout.shape[0]
'''
