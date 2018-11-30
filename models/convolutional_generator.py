import torch.nn as nn
from constants import *
import random


class ConvEncoderRNNDecoder(nn.Module):
    def __init__(self, encoder, decoder, max_length=MAX_LENGTH, criterion=nn.CrossEntropyLoss(ignore_index=0)):

        super(ConvEncoderRNNDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.criterion = criterion

    def forward(self, input_tensor, target_tensor):

        batch_size = input_tensor.shape[0]
        input_length = input_tensor.shape[1]

        target_length = target_tensor.shape[1]

        loss = 0

        encoder_outputs = torch.zeros(self.max_length, batch_size, self.encoder.hidden_size, device=DEVICE)
        encoder_hidden = self.encoder.forward(input_tensor).transpose(0, 1)

        for ei in range(input_length):
            encoder_outputs[ei, :, :] = encoder_hidden[ei, :, :]

        decoder_input = torch.tensor([[SOS_INDEX] * batch_size], device=DEVICE).transpose(0, 1)
        decoder_hidden = encoder_hidden[-1, :, :].unsqueeze(0)

        use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[:, di])
                decoder_input = target_tensor[:, di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[:, di])

        return loss
