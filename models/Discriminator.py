import torch
import torch.nn as nn
from constants import DEVICE, SOS_INDEX


class Discriminator(nn.Module):

	def __init__(self, encoder, decoder, hidden_size, vocab_size, num_layers):
		super(Discriminator, self).__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.vocab_size = vocab_size
		self.sigmoid = nn.Sigmoid()
		self.num_layers = num_layers

	def forward(self, context_tensor, input_tensor, true_sample):

		batch_size = input_tensor.shape[0]
		original_input_length = input_tensor.shape[1]

		# Use the context as input for the encoder
		encoder_input = input_tensor
		encoder_input_length = encoder_input.shape[1]

		encoder_hidden = self.encoder.initHidden(batch_size=batch_size, num_layers=self.num_layers)
		encoder_outputs = torch.zeros(encoder_input_length, batch_size, self.encoder.hidden_size, device=DEVICE)

		# Perform the encoder steps for each word in the input tensor
		for ei in range(encoder_input_length):
			encoder_output, encoder_hidden = self.encoder(encoder_input[:, ei], encoder_hidden)
			encoder_outputs[ei, :, :] = encoder_output[0, :, :]

		# # Initialize the decoder input and hidden state, and an empty tensor to store the output values
		# decoder_input = torch.tensor([[SOS_INDEX] * batch_size], device=DEVICE).view(batch_size, 1)
		# decoder_hidden = encoder_hidden
		# decoder_outputs = torch.zeros(batch_size, original_input_length, device=DEVICE)

		# # Perform the decoder steps for as many steps as there are words in the given generated/true reply
		# for di in range(original_input_length):
		# 	decoder_output, decoder_hidden, _ = self.decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
		# 	# decoder_output = self.sigmoid(decoder_output)
		# 	decoder_output - decoder_output[target_index]
		# 	decoder_outputs[:, di] = decoder_output.squeeze(dim=1)

		# Initialize the decoder input and hidden state, and an empty tensor to store the output values
		decoder_input = torch.tensor([[SOS_INDEX] * batch_size], device=DEVICE).view(batch_size, 1)
		decoder_hidden = encoder_hidden
		decoder_outputs = torch.zeros(batch_size, original_input_length, self.vocab_size, device=DEVICE)

		# Perform the decoder steps for as many steps as there are words in the given generated/true reply
		for di in range(original_input_length):
			decoder_output, decoder_hidden, _ = self.decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
			# decoder_output = self.sigmoid(decoder_output)
			decoder_outputs[:, di, :] = decoder_output

		# Interpret the decoder output as probabilities per word in the vocabulary, 
		# and select the probabilities of the words in the given generated/true reply
		out_probabilities = torch.zeros(input_tensor.shape, device=DEVICE)
		for batch in range(decoder_outputs.shape[0]):
			for word in range(decoder_outputs.shape[1]):
				out_probabilities[batch, word] = decoder_outputs[batch, word, input_tensor[batch, word].item()]

		# word_level_rewards = -torch.log(out_probabilities)
		word_level_rewards = out_probabilities
		sent_level_rewards = torch.mean(word_level_rewards, dim=1)
		avg_batch_reward = torch.mean(sent_level_rewards, dim=0)

		# Compute a loss value using the average reward and the target tensor
		if true_sample:
			loss = -avg_batch_reward
		else:
			loss = avg_batch_reward
		
		return loss, word_level_rewards, sent_level_rewards