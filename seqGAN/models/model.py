import torch
import torch.nn as nn
import torch.autograd as autograd
from torch import optim
import torch.nn.functional as F
from constants import *
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, encoder, decoder, embedding_dim, hidden_dim, vocab_size, max_length=MAX_LENGTH, criterion=nn.NLLLoss()):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.criterion = criterion

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

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

        use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

        decoder_outputs = []
        words = []

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[:, di])
                sampled_word = torch.multinomial(torch.exp(decoder_output), 1)
                decoder_input = target_tensor[:, di]  # Teacher forcing
                decoder_outputs.append(decoder_output)
                words.append(sampled_word)

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[:, di])
                sampled_word = torch.multinomial(torch.exp(decoder_output), 1)
                decoder_outputs.append(decoder_output)
                words.append(sampled_word)

        words = torch.stack(words)
        decoder_outputs = torch.stack(decoder_outputs)
        return loss, decoder_outputs, words

    def sample(self, input_tensor, start_letter=0, length=None):

        if not length:
            length = self.max_length
        if isinstance(input_tensor[0], tuple):
            input_tensor = [tensor for (tensor, target) in input_tensor]
        input_tensor = torch.nn.utils.rnn.pad_sequence(input_tensor, batch_first=True)

        batch_size = input_tensor.shape[0]
        input_length = input_tensor.shape[1]

        encoder_hidden = self.encoder.initHidden(batch_size=batch_size).to(DEVICE)

        encoder_outputs = torch.zeros(self.max_length, batch_size, self.encoder.hidden_size, device=DEVICE)

        loss = 0

        input_tensor = input_tensor.to(DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei, :, :] = encoder_output[0, :, :]

        decoder_input = torch.tensor([[SOS_INDEX] * batch_size], device=DEVICE).transpose(0, 1)

        decoder_hidden = encoder_hidden

        decoder_outputs = []

        # Without teacher forcing: use its own predictions as the next input
        for di in range(length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            # decoder_input = topi.squeeze().detach()  # detach from history as input
            sampled_word = torch.multinomial(torch.exp(decoder_output), 1)
            decoder_input = sampled_word.detach()
            # torch.multinomial(torch.exp(decoder_output), 1)
            decoder_outputs.append(sampled_word)

        decoder_outputs = torch.stack(decoder_outputs)

        return decoder_outputs

    def initHidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_dim, device=DEVICE)


    def PG_forward(self, inp, hidden):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.gru(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        inp = inp.transpose(0,1).squeeze(-1)
        batch_size, seq_len = inp.size()
        # inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        h = self.initHidden(batch_size=batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.PG_forward(inp[:, i], h)
            # TODO: should h be detached from graph (.detach())?

            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size


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

        output = self.log_softmax(self.out(output))
        output = output.squeeze(0)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
