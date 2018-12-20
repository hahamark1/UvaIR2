import torch
import torch.nn as nn
from constants import *
import torch.nn.functional as F


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH, num_layers=1, LSTM='LSTM'):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        if LSTM == 'LSTM':
            self.gru = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers)
        elif LSTM == 'GRU':
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers)
        else:
            assert "This is not a correct type of RNN cell."
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.device = DEVICE
        self.LSTM = LSTM

        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, encoder_outputs):
        batch_size = input.shape[0]
        embedded = self.embedding(input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)

        if self.LSTM == 'LSTM':
            if self.num_layers > 1:
                concat = torch.cat((embedded, hidden[0][-1].unsqueeze(0)), 2)
            else:
                concat = torch.cat((embedded, hidden[0]), 2)
        else:
            if self.num_layers > 1:
                concat = torch.cat((embedded, hidden[-1].unsqueeze(0)), 2)
            else:
                concat = torch.cat((embedded, hidden), 2)

        concat = self.attn(concat)
        attn_weights = F.softmax(concat, dim=2)

        attn_weights = attn_weights.transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        attn_applied = attn_applied.transpose(0, 1)

        output = torch.cat((embedded, attn_applied), 2)

        output = self.attn_combine(output)

        output = self.relu(output)

        output, hidden = self.gru(output, hidden.contiguous())

        output = self.out(output)
        output = self.log_softmax(output)
        output = output.squeeze(0)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


if __name__ == '__main__':

    ADR = AttnDecoderRNN(512, 1000)
