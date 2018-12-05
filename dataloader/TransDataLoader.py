import unicodedata
import re
import random

import torch

from constants import MAX_LENGTH
from Language import Language


class TransDataLoader():

    def __init__(self, input_lang_name, output_lang_name, device='cpu'):
        self.input_lang_name = input_lang_name
        self.output_lang_name = output_lang_name
        self.device = device
        self.input_lang = None
        self.output_lang = None
        self.pairs = None
        self.SOS_token = 0
        self.EOS_token = 1
        self.readData()

    def readData(self):
        print("Reading lines...")

        # Read the file and split into lines
        lines = open('../data/%s-%s/%s-%s.txt' % (self.input_lang_name, self.output_lang_name, self.input_lang_name, self.output_lang_name), encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        self.pairs = [[self.normalizeString(s) for s in l.split('\t')] for l in lines]
        self.pairs = self.filterPairs(self.pairs)

        self.input_lang = Language(self.input_lang_name)
        self.output_lang = Language(self.output_lang_name)
        for pair in self.pairs:
            self.input_lang.addSentence(pair[0])
            self.output_lang.addSentence(pair[1])

    def getListOfTensors(self, n_pairs):
        return [self.pairToTensor(random.choice(self.pairs)) for i in range(n_pairs)]

    def pairToTensor(self, pair):
        input_tensor = self.tensorFromSentence(self.input_lang, pair[0])
        target_tensor = self.tensorFromSentence(self.output_lang, pair[1])
        return (input_tensor, target_tensor)

    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

    def unicodeToAscii(self, s):
        return ''.join(
        	c for c in unicodedata.normalize('NFD', s)
        	if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def filterPair(self, p):
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH

    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]
