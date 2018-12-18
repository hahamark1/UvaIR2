import torch
import torch.nn as nn
from constants import *


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, output_size, num_layers=1, LSTM='LSTM'):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if LSTM == 'LSTM':
            self.gru = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers)
        elif LSTM == 'GRU':
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers)
        else:
            assert "This is not a correct type of RNN cell."
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = DEVICE
        self.relu = nn.ReLU()

    def forward(self, input, hidden, encoder_outputs):
        batch_size = input.shape[0]
        output = self.embedding(input).view(1, batch_size, -1)
        output = self.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden, None

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
