# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:41:38 2018

@author: Joris
"""
import unicodedata
import re
import os
import torch
import string
from constants import *
import torch
from torch.utils.data import Dataset, DataLoader

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOU_token]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensorsFromPair(pair, language):
    input_tensor = tensorFromSentence(language, pair[0])
    target_tensor = tensorFromSentence(language, pair[1])
    return (input_tensor, target_tensor)


class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>": 0, "<EOS>":1, "__eou__": 2}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "__eou__"}
        self.n_words = 3  # Count SOS, EOS and __eou__

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            try:
                self.word2count[word] += 1
            except:
                pass


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters. Furthermore, remove double
# spaces
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub('[%s]' % re.escape(string.punctuation), '', s)
    s = s.strip()
    s = ' '.join(s.split())

    return s

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def readLangs(filename, filter_=True):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(os.path.join('datasets', filename), encoding='utf-8').\
        read().strip().split('\n')
                
    # Split every line into pairs and normalize
    utterences = [[normalizeString(s) for s in l.split("__eou__")] for l in lines]
    
    Language = Lang(name='language')
    
    dialogues = []
    
    for utterence in utterences:
        
        conversation = []
        for line in utterence:
            conversation.append(line.strip())
            
        try: 
            conversation.remove('')
            conversation.remove(' ')
        except:
            pass
            
        questions = conversation[:-1]
        answer = conversation[-1]
            
        dialogues.append((" __eou__ ".join(x for x in questions), answer))
        
    if filter_:
        dialogues = filterPairs(dialogues)
        
    for q, a in dialogues:
        Language.addSentence(q)
        Language.addSentence(a)
    
    print('Starting to train with {} dialogue pairs'.format(len(dialogues)))
    print('The vocab size is {}'.format(Language.n_words))
 
    return Language, dialogues

