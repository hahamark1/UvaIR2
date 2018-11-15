import torch
from torch.utils.data import Dataset
import unicodedata
import string
import re

import sys
sys.path.append("..")
from constants import PADDING_TOKEN, SPLIT_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, NUM_TOKEN, CONTRACTIONS


N_UTTERANCES_FOR_INPUT = 3


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
	s = ' '.join(s.split())
	return s.lower()


class Vocabulary():

	def __init__(self):

		self.padding_token = PADDING_TOKEN
		self.split_token = SPLIT_TOKEN
		self.unk_token = UNK_TOKEN
		self.sos_token = SOS_TOKEN
		self.eos_token = EOS_TOKEN
		self.num_token = NUM_TOKEN

		self.word2index = {self.padding_token: 0, self.split_token: 1, self.unk_token: 2, self.sos_token: 3, self.eos_token: 4, self.num_token: 5}
		self.word2count = {}
		self.index2word = {0: self.padding_token, 1: self.split_token, 2: self.unk_token, 3: self.sos_token, 4: self.eos_token, 5:self.num_token}
		self.n_words = 6

	def add_word(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

	def add_utterance(self, utterance):
		for word in utterance:
			self.add_word(word)

	def add_dialogue(self, dialogue):
		for utterance in dialogue:
			self.add_utterance(utterance)


class DailyDialogLoader(Dataset):

	def __init__(self, path_to_data):
		self.path_to_data = path_to_data

		# Initalize a Vocabulary object
		self.vocabulary = Vocabulary()

		# Initialize the lists in which to store the good stuff
		self.dialogues = []
		self.inputs, self.targets = [], []
		self.n_utterances_for_input = N_UTTERANCES_FOR_INPUT


		# Execute the functions to load the data
		self.read_txt()
		self.fill_vocabulary()
		self.convert_to_onehot()
		self.split_inputs_targets()

	def __len__(self):
		# Needed for the PyTorch DataLoader, returns the length of the dataset
		return len(self.inputs)

	def __getitem__(self, idx):
		# Needed for the PyTorch DataLoader, returns an [input, target] pair given an index
		return self.inputs[idx], self.targets[idx]

	def read_txt(self):
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

				words = normalizeString(line).split(' ')

				# Find all the indices of the split token __eou__ (end of utterance)
				split_indices = [i for i, x in enumerate(words) if x == '__eou__' or x == '__eou__\n']
				# Add a -1 in front of this list
				split_indices.insert(0, -1)

				# For each index, get the split of the words list until that index (a bit of magic with +1's to get rid of the split token)
				utterances = [words[x + 1:split_indices[i + 1]] for i, x in enumerate(split_indices[:-1])]
				dialogues.append(utterances)

		self.dialogues = dialogues

	def fill_vocabulary(self):
		# Fill the vocabulary of the Vocabulary class
		for dialogue in self.dialogues:
			self.vocabulary.add_dialogue(dialogue)

	def convert_to_onehot(self):
		# For each word in all dialogues, convert it to its index in the onehot encoding
		self.dialogues = [[[self.word_to_onehot(word) for word in utterance] for utterance in dialogue] for dialogue in self.dialogues]

	def word_to_onehot(self, word):
		return self.vocabulary.word2index[word]

	def split_inputs_targets(self):
		# Splits the dialogues into inputs and targets
		# Uses the constant N_UTTERANCES_FOR_INPUT to determine how many utterances should make up an input
		# The target always consists of one utterance

		inputs = []
		targets =[]
		for dialogue in self.dialogues:
			# If the utterance is not long enough to split into a train and target, just skip it
			if len(dialogue) <= self.n_utterances_for_input+1:
				continue

			# i is the start index of the input/target pair
			for i in range(len(dialogue) - self.n_utterances_for_input):
				input_combination = []
				for j in range(self.n_utterances_for_input):
					input_combination = input_combination + dialogue[i+j]
					if j != self.n_utterances_for_input - 1:
						input_combination.append(self.vocabulary.word2index[self.vocabulary.split_token])
				inputs.append(torch.tensor(input_combination))
				targets.append(torch.tensor(dialogue[i+self.n_utterances_for_input]))

		self.inputs = inputs
		self.targets = targets


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

	DDL = DailyDialogLoader('../data/dailydialog/train/dialogues_train.txt')
	dataloader = DataLoader(DDL, batch_size=15, shuffle=True, num_workers=0, collate_fn=PadCollate(pad_front=False))