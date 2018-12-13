import torch.nn as nn
from constants import *
import random


class ConvDiscriminator(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, max_length=MAX_LENGTH, num_layers=NUM_LAYERS, criterion=nn.CrossEntropyLoss(ignore_index=0)):

        super(ConvDiscriminator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.sigmoid = nn.Sigmoid()
        self.loss_fnc = nn.BCELoss()

    def forward(self, input_tensor, target_tensor, true_sample):

        batch_size = input_tensor.shape[0]
        input_length = input_tensor.shape[1]

        target_length = target_tensor.shape[1]
        loss = 0

        encoder_outputs = torch.zeros(self.max_length, batch_size, self.encoder.hidden_size, device=DEVICE)
        encoder_hidden = self.encoder.forward(input_tensor).transpose(0, 1)

        for ei in range(input_length):
            encoder_outputs[ei + (self.max_length - input_length), :, :] = encoder_hidden[ei, :, :]

        decoder_input = torch.tensor([[SOS_INDEX] * batch_size], device=DEVICE).transpose(0, 1)
        decoder_hidden = encoder_hidden[-1, :, :]
        decoder_hidden = torch.stack([decoder_hidden] * self.num_layers, 0)
        decoder_outputs = torch.zeros(batch_size, target_length, self.vocab_size, device=DEVICE)

        decoder_input = decoder_input.contiguous()
        decoder_hidden = decoder_hidden.contiguous()
        # Perform the decoder steps for as many steps as there are words in the given generated/true reply
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = self.decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
            decoder_output = self.sigmoid(decoder_output)
            decoder_outputs[:, di, :] = decoder_output

        # Interpret the decoder output as probabilities per word in the vocabulary,
        # and select the probabilities of the words in the given generated/true reply
        out_probabilities = torch.zeros(input_tensor.shape, device=DEVICE)
        for batch in range(decoder_outputs.shape[0]):
            for word in range(decoder_outputs.shape[1]):
                out_probabilities[batch, word] = decoder_outputs[batch, word, input_tensor[batch, word].item()]

        # Create a target tensor of either zeros or ones depending whether the reply is true or generated
        target = torch.ones(input_tensor.shape, device=DEVICE) * int(true_sample)

        # Compute a loss value using the selected probabilities and the target tensor
        loss = self.loss_fnc(out_probabilities, target)

        loss /= batch_size

        return loss, out_probabilities
