import torch
import torch.nn as nn
from constants import *


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = DEVICE
        self.relu = nn.ReLU()

    def forward(self, input, hidden, encoder_outputs):
        if hidden is None:
            hidden = self.initHidden()

        batch_size = input.shape[0]
        output = self.embedding(input).view(1, batch_size, -1)
        output = self.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden, None

    def initHidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)