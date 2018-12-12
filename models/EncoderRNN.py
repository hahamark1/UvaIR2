import torch.nn as nn
import torch
from constants import *


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, LSTM='LSTM'):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        if LSTM == 'LSTM':
            self.gru = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers)
        elif LSTM == 'GRU':
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers)
        else:
            assert "This is not a correct type of RNN cell."
        self.device = DEVICE

    def forward(self, input, hidden):
        batch_size = input.shape[0]
        embedded = self.embedding(input).view(1, batch_size, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size=1, num_layers=1):
            return torch.zeros(num_layers, batch_size, self.hidden_size, device=self.device)
