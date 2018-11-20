import torch
import torch.nn as nn
from constants import DEVICE, SOS_INDEX


class Discriminator(nn.Module):

	def __init__(self, encoder, decoder, vocab_size):
		super(Discriminator, self).__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.loss_fnc = nn.NLLLoss()
		self.vocab_size = vocab_size

	def forward(self, context_tensor, input_tensor, true_sample):

		batch_size = input_tensor.shape[0]
		original_input_length = input_tensor.shape[1]

		encoder_input = torch.cat((context_tensor, input_tensor), 1)
		encoder_input_length = encoder_input.shape[1]

		encoder_hidden = self.encoder.initHidden(batch_size=batch_size)
		encoder_outputs = torch.zeros(encoder_input_length, batch_size, self.encoder.hidden_size, device=DEVICE)

		for ei in range(encoder_input_length):
			encoder_output, encoder_hidden = self.encoder(encoder_input[:, ei], encoder_hidden)
			encoder_outputs[ei, :, :] = encoder_output[0, :, :]

		decoder_input = torch.tensor([[SOS_INDEX] * batch_size], device=DEVICE).view(batch_size, 1)
		decoder_hidden = encoder_hidden
		decoder_outputs = torch.zeros(batch_size, self.vocab_size, original_input_length, device=DEVICE)

		loss = 0

		for di in range(original_input_length):
			decoder_output, decoder_hidden, _ = self.decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
			decoder_outputs[:, :, di] = decoder_output
			
			# Determine the top outputs of the decoder
			topv, topi = decoder_output.topk(1)
			target = topi

			if true_sample:
				loss += self.loss_fnc(decoder_output, target)
			else:
				loss += self.loss_fnc()

		return loss, decoder_outputs