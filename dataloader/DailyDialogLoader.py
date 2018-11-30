import torch
from torch.utils.data import Dataset
import numpy as np
import unicodedata
import string
import re
from collections import defaultdict
import os
import pickle
import sys
sys.path.append("..")
from constants import *


def unicodeToAscii(s):
	"""
	:param s: a string, example 'Joris is the beste'
	:return: string in Ascii format
	"""
	return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
	"""
	:param s: string
	:return: string s
	"""
	s = unicodeToAscii(s.lower().strip())

	s = re.sub("’", "'", s)
	s = re.sub(" ’ ", "'", s)
	s = re.sub(" ' ", "'", s)

	# Replace contractions
	for word in s.split():
		if word.lower() in CONTRACTIONS:
			s = s.replace(word, CONTRACTIONS[word.lower()])

	# Remove punctuation
	exclude = set(string.punctuation)
	exclude.add("’")
	exclude.add("£")
	exclude.add("$")
	s = ''.join(ch for ch in s if ch not in exclude)

	# Remove double spaces
	s = re.sub("  ", " ", s)

	# Replace with __eou__
	s = re.sub(" eou", " __eou__", s)

	# Replace digits
	s = re.sub('\d+', NUM_TOKEN, s)

	# We do not want to take the last __eou__ into account
	s = ' '.join(s.split()[:-1])
	s = re.sub('__eou__ __eou__', '__eou__', s)
	return s.lower()

def filterPair(p):
    return len(p[0].split(' ')) < MAX_UTTERENCE_LENGTH and len(p[1].split(' ')) < MAX_UTTERENCE_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


class Vocabulary():

	def __init__(self, load):

		self.padding_token = PADDING_TOKEN
		self.split_token = SPLIT_TOKEN
		self.unk_token = UNK_TOKEN
		self.sos_token = SOS_TOKEN
		self.eos_token = EOS_TOKEN
		self.num_token = NUM_TOKEN
		self.eou_token = EOU_TOKEN

		self.word2index = {PADDING_TOKEN: PADDING_INDEX, SPLIT_TOKEN: SPLIT_INDEX, UNK_TOKEN: UNK_INDEX, SOS_TOKEN: SOS_INDEX, EOS_TOKEN: EOS_INDEX, NUM_TOKEN: NUM_INDEX, EOU_TOKEN: EOU_INDEX}
		self.word2count = {key: 0 for key in self.word2index.keys()}
		self.index2word = {self.word2index[key]: key for key in self.word2index.keys()}
		self.n_words = 7

		self.word2index = defaultdict(lambda: UNK_INDEX, self.word2index)
		self.index2word = defaultdict(lambda: UNK_TOKEN, self.index2word)
		self.word2count = defaultdict(lambda: 0, self.word2count)

		self.loaded = False

		if load:
			self.load_vocabulary()
			self.loaded = True
			print('Loaded the dictionary')


	def add_word(self, word):

		if self.loaded:
			return

		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

	def add_utterance(self, utterance):

		if self.loaded:
			return

		utterance = utterance.split(' ')
		for word in utterance:
			self.add_word(word)

	def tokens_to_sent(self, tensor):
		tensor_list = tensor.cpu().data.numpy().tolist()
		sentence = [self.index2word[token] for token in tensor_list]
		sentence = [word for word in sentence if word != self.padding_token]
		return " ".join(sentence)

	def list_to_sent(self, list):
		sentence = [self.index2word[token] for token in list]
		sentence = [word for word in sentence if word != self.padding_token]
		return " ".join(sentence)

	def save_vocabulary(self):
		""" Save the vocabulary and word2index and index2word """

		self.word2index = dict(self.word2index)
		self.index2word = dict(self.index2word)

		with open(os.path.join(PATH_TO_SAVE, 'word2index_index2word.p'), 'wb') as handle:
			pickle.dump((self.word2index, self.index2word), handle)

		print('Wrote the vocabulary to a pickle')

	def load_vocabulary(self):
		""" Load a saved vocabulary and word2index and index2word"""

		with open(os.path.join(PATH_TO_SAVE, 'word2index_index2word.p'), 'rb') as handle:
			self.word2index, self.index2word = pickle.load(handle)

		self.word2index = defaultdict(lambda: UNK_INDEX, self.word2index)
		self.index2word = defaultdict(lambda: UNK_TOKEN, self.index2word)
		self.n_words = len(self.index2word.keys())

		assert len(self.word2index) == len(self.index2word)

class DailyDialogLoader(Dataset):

	def __init__(self, path_to_data, load=False):
		self.path_to_data = path_to_data

		# Initalize a Vocabulary object
		self.vocabulary = Vocabulary(load=load)

		# Initialize the lists in which to store the good stuff
		self.dialogues = []
		self.inputs, self.targets = [], []
		self.n_utterances_for_input = N_UTTERANCES_FOR_INPUT

		self.eou_token = EOU_TOKEN
		self.sos_token = SOS_TOKEN
		self.eos_token = EOS_TOKEN

		# Init the self.dialogues
		self.dialogues = []

		# Execute the functions to load the data
		self.read_txt(utterance_length=N_UTTERANCES_FOR_INPUT)
		self.read_txt(utterance_length=N_UTTERANCES_FOR_INPUT+1)
		self.read_txt(utterance_length=N_UTTERANCES_FOR_INPUT+2)
		self.read_txt(utterance_length=N_UTTERANCES_FOR_INPUT+3)
		self.fill_vocabulary()
		self.convert_to_onehot()
		self.split_inputs_targets()

		self.max_words = np.max([len(x) for x in self.inputs])

		print('Starting to train with {} dialogue pairs'.format(len(self.dialogues)))
		print('The vocab size is {}'.format(self.vocabulary.n_words))

		if not load:
			self.vocabulary.save_vocabulary()

	def __len__(self):
		# Needed for the PyTorch DataLoader, returns the length of the dataset
		return len(self.inputs)

	def __getitem__(self, idx):
		# Needed for the PyTorch DataLoader, returns an [input, target] pair given an index
		return self.inputs[idx], self.targets[idx]

	def read_txt(self, utterance_length=N_UTTERANCES_FOR_INPUT):
		# Reads a txt file, returns a list of dialogues, consisting of lists of utterances, consisting of words
		# Example
		# 		-the whole thing consists of two dialogues
		# 		-the first dialogue consists of two utterances, the second one of one utterance
		# 		-each line consists of one utterance
		# dialogues = [[[word1, word2, word3],
		# 			  [word2, word4, word1]],
		# 		     [[word5, word6, word7]]]

		with open(self.path_to_data) as f:

			lines = f.readlines()
			# Keep a list of the dialogues
			dialogues = []

			for line in lines:

				words = normalizeString(line)

				utterances = words.split(' {} '.format(self.eou_token))

				if len(utterances) < utterance_length+1:
					continue

				for index in range(0, len(utterances)-utterance_length - 1):

					question = ' {} '.format(self.eou_token).join(utterances[index:index+utterance_length])
					question = '{} {}'.format(self.sos_token, question).strip()

					target = utterances[index + utterance_length + 1]
					target = '{} {}'.format(target, self.eos_token).strip()

					dialogues.append((question, target))

		dialogues = filterPairs(dialogues)

		self.dialogues += dialogues

	def fill_vocabulary(self):
		# Fill the vocabulary of the Vocabulary class
		for question, answer in self.dialogues:
			self.vocabulary.add_utterance(question)
			self.vocabulary.add_utterance(answer)

	def convert_to_onehot(self):
		# For each word in all dialogues, convert it to its index in the onehot encoding
		self.one_hot_dialogues = []

		for question, answer in self.dialogues:

			question = [self.word_to_onehot(word) for word in question.split(' ')]
			answer = [self.word_to_onehot(word) for word in answer.split(' ')]

			self.one_hot_dialogues.append((question, answer))

	def word_to_onehot(self, word):
		return self.vocabulary.word2index[word]

	def split_inputs_targets(self):
		# Splits the dialogues into inputs and targets
		# Uses the constant N_UTTERANCES_FOR_INPUT to determine how many utterances should make up an input
		# The target always consists of one utterance

		self.inputs = [torch.tensor(question) for question, _ in self.one_hot_dialogues]
		self.targets = [torch.tensor(answer) for _, answer in self.one_hot_dialogues]

# Based on: https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/7
class PadCollate:
	"""
	a variant of collate_fn that pads according to the longest sequence in
	a batch of sequences
	"""

	def __init__(self, pad_front=True):
		self.pad_front = pad_front

	def pad_collate(self, batch):
		"""
		args:
		    batch - list of [input, target]

		return:
		    inputs - a tensor of all inputs in batch after padding
		    targets - a tensor of all targets in batch after padding
		"""

		# Find longest sequence
		max_input_length = max([len(input_target_pair[0]) for input_target_pair in batch])
		max_target_length = max([len(input_target_pair[1]) for input_target_pair in batch])

		# Pad 'm
		batch = [[self.pad_tensor(input_target_pair[0], max_input_length), \
				 self.pad_tensor(input_target_pair[1], max_target_length)] \
					for input_target_pair in batch]

		# Stack the inputs together and the targets together
		inputs = torch.stack([input_target_pair[0] for input_target_pair in batch])
		targets = torch.stack([input_target_pair[1] for input_target_pair in batch])

		return inputs, targets

	def pad_tensor(self, vec, pad):
		"""
		args:
		    vec - tensor to pad
		    pad - the size to pad to

		return:
		    a new tensor padded to 'pad' in dimension 'dim'
		"""
		pad_size = list(vec.shape)
		pad_size[0] = pad - vec.size(0)

		vec = vec.type(torch.LongTensor)

		if self.pad_front:
			return torch.cat([torch.zeros(*pad_size).type(torch.LongTensor), vec], dim=0)
		else:
			return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=0)


	def __call__(self, batch):
		return self.pad_collate(batch)


if __name__ == '__main__':

	from torch.utils.data import Dataset, DataLoader

	PATH_TO_SAVE = os.path.join('..', 'saved_models')
	DDL = DailyDialogLoader('../data/dailydialog/train/dialogues_train.txt')
	dataloader = DataLoader(DDL, batch_size=16, shuffle=True, num_workers=0, collate_fn=PadCollate(pad_front=True))

	for i, (data, target) in enumerate(dataloader):

		for j in range(0, target.shape[0]):
			continue
			#print(dataloader.dataset.vocabulary.tokens_to_sent(data[j]))
			#print(dataloader.dataset.vocabulary.tokens_to_sent(target[j]))
			#print('')
