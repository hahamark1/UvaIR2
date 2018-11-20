import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from constants import *
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, encoder, decoder, max_length=MAX_LENGTH, criterion=nn.NLLLoss()):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.criterion = criterion

    def forward(self, input_tensor, target_tensor):
        batch_size = input_tensor.shape[0]
        input_length = input_tensor.shape[1]

        target_length = target_tensor.shape[1]

        encoder_hidden = self.encoder.initHidden(batch_size=batch_size)
        encoder_outputs = torch.zeros(self.max_length, batch_size, self.encoder.hidden_size, device=DEVICE)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei, :, :] = encoder_output[0, :, :]

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
        
        generator_output = generator_output.view(batch_size, target_length)
        return loss, generator_output

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = DEVICE

    def forward(self, input, hidden):
        batch_size = input.shape[0]
        embedded = self.embedding(input).view(1, batch_size, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = DEVICE
        self.relu = nn.ReLU()

    def forward(self, input, hidden, encoder_outputs):
        batch_size = input.shape[0]
        output = self.embedding(input).view(1, batch_size, -1)
        output = self.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden, None

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.device = DEVICE

        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, hidden, encoder_outputs):
        batch_size = input.shape[0]
        embedded = self.embedding(input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)

        concat = torch.cat((embedded, hidden), 2)
        concat = self.attn(concat)
        attn_weights = F.softmax(concat, dim=2)

        attn_weights = attn_weights.transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        attn_applied = attn_applied.transpose(0 ,1)

        output = torch.cat((embedded, attn_applied), 2)

        output = self.attn_combine(output)

        output = self.relu(output)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output))
        output = output.squeeze(0)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
