import torch.nn as nn
from constants import *
import random


class Generator(nn.Module):
    def __init__(self, encoder, decoder, LSTM='LSTM', num_layers=1, max_length=MAX_LENGTH, criterion=nn.NLLLoss()):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.criterion = criterion
        self.num_layers = num_layers
        self.LSTM = LSTM

    def forward(self, input_tensor, target_tensor):

        batch_size = input_tensor.shape[0]
        input_length = input_tensor.shape[1]

        target_length = target_tensor.shape[1]
        if self.LSTM == 'LSTM':
            encoder_hidden = (self.encoder.initHidden(batch_size=batch_size, num_layers=self.num_layers),
                                self.encoder.initHidden(batch_size=batch_size, num_layers=self.num_layers))
        else:
            encoder_hidden = self.encoder.initHidden(batch_size=batch_size, num_layers=self.num_layers)

        encoder_outputs = torch.zeros(self.max_length, batch_size, self.encoder.hidden_size, device=DEVICE)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei + self.max_length - input_length, :, :] = encoder_output[0, :, :]

        decoder_input = torch.tensor([[SOS_INDEX] * batch_size], device=DEVICE).transpose(0, 1)

        decoder_hidden = encoder_hidden
        generator_output = torch.zeros(target_length, batch_size, device=DEVICE).long()

        use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[:, di])
                decoder_input = target_tensor[:, di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                generator_output[di, :] = topi.view(-1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[:, di])

        generator_output = generator_output.permute(1, 0)
        return loss, generator_output

    def generate_sentence(self, context_tensor):
        
        context_length = context_tensor.shape[1]      

        # Initialize an empty tensor for the outputs of the encoder
        encoder_outputs = torch.zeros(self.max_length, 1, self.encoder.hidden_size, device=DEVICE)
        encoder_hidden = None

        for ei in range(context_length):
            encoder_output, encoder_hidden = self.encoder(context_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei + self.max_length - context_length, :, :] = encoder_output[0, :, :]

        decoder_input = torch.tensor([[SOS_INDEX]], device=DEVICE).view(-1, 1)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        generator_output = torch.zeros(MAX_WORDS_GEN, device=DEVICE).long()

        for di in range(MAX_WORDS_GEN):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            topv, topi = decoder_output.data.topk(1)
            generator_output[di] = topi.view(-1)

            if topi.item() == EOS_INDEX:
                decoded_words.append(EOS_INDEX)
                generator_output = generator_output[:di+1]
                break
            else:
                decoded_words.append(topi.item())
            
            decoder_input = topi.detach()

        return decoded_words, generator_output

if __name__ == '__main__':
    exit()
