import torch
import torch.nn as nn
from constants import DEVICE


class Discriminator(nn.Module):

	def __init__(self, input_size, hidden_size):
		super(Discriminator, self).__init__()

		self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.criterion = criterion

	def forward(self, context_tensor, input_tensor):

		print('context shape', context_tensor.shape)
		print('input shape', input_tensor.shape)
		encoder_input = torch.cat((context_tensor, input_tensor), 1)
		print('concatenated shape', encoder_input.shape)
		
		batch_size = input_tensor.shape[0]
		input_length = encoder_input.shape[1]

		encoder_hidden = self.encoder.initHidden(batch_size=batch_size)
		encoder_outputs = torch.zeros(self.max_length, batch_size, self.encoder.hidden_size, device=DEVICE)

		for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(encoder_input[:, ei], encoder_hidden)
            encoder_outputs[ei, :, :] = encoder_output[0, :, :]