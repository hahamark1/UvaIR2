# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:56:42 2018

@author: Joris
"""

import torch
from models.model import AttnDecoderRNN, EncoderRNN, DecoderRNN, Generator
import random
import time
from utils.seq2seq_helper_funcs import showPlot, asMinutes, timeSince
import torch.optim as optim
import torch.nn as nn
from constants import *
from dataloader.DailyDialogLoader import DailyDialogLoader, PadCollate
from torch.utils.data import Dataset, DataLoader
import os

teacher_forcing_ratio = 0.5

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, question=False):
    indexes = indexesFromSentence(lang, sentence)

    if question:
        indexes.append(EOU_token)
    else:
        indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensorsFromPair(pair, language):
    input_tensor = tensorFromSentence(language, pair[0], question=True)
    target_tensor = tensorFromSentence(language, pair[1], question=False)
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, generator, optimizer, criterion, max_length=MAX_LENGTH):

    optimizer.zero_grad()
    loss = generator(input_tensor, target_tensor)
    loss.backward()
    optimizer.step()
    target_length = target_tensor.shape[1]

    return loss.item() / target_length


def trainIters(generator, dataloader, num_epochs=3000, plot_every=100,
               evaluate_every=100, save_every=100, learning_rate=0.01):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=0, size_average=False)

    num_iters = len(dataloader)
    n_iters = 0

    for epoch in range(num_epochs):

        for i, (input_tensor, target_tensor) in enumerate(dataloader):


            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)

            loss = train(input_tensor, target_tensor, generator, optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss

            if n_iters % plot_every == 0 and n_iters > 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            if n_iters % evaluate_every == 0:
                test_sentence = input_tensor[0, :]
                test_target_sentence = target_tensor[0, :]

                real_test_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_sentence)
                real_target_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_target_sentence)

                decoded_words = evaluate(generator.encoder, generator.decoder, test_sentence)
                generated_sentence = dataloader.dataset.vocabulary.list_to_sent(decoded_words)

                print(real_test_sentence)
                print('>>')
                print(generated_sentence)
                print('==')
                print(real_target_sentence)
                print('-----------------------------')

            if n_iters % save_every == 0 and n_iters > 0:
                torch.save(generator, os.path.join('saved_models', 'generator.pt'))

            n_iters += 1

        scheduler.step()

        plot_loss_avg = plot_loss_total / num_iters
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
        print('After epoch {} the loss is {}'.format(epoch, plot_loss_avg))

    showPlot(plot_losses)

def evaluate(encoder, decoder, input_tensor, max_length=MAX_LENGTH):
    with torch.no_grad():

        input_tensor = input_tensor.view(1, -1)
        input_length = input_tensor.shape[1]

        encoder_outputs = torch.zeros(max_length, 1, encoder.hidden_size, device=DEVICE)
        encoder_hidden = None

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei, :, :] = encoder_output[0, :, :]

        decoder_input = torch.tensor([[SOS_INDEX]], device=DEVICE).view(-1, 1)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(MAX_WORDS_GEN):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_INDEX:
                decoded_words.append(EOS_INDEX)
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.detach()

        print(decoded_words)
        return decoded_words

if __name__ == '__main__':

    PATH_TO_DATA =  'data/dailydialog/train/dialogues_train.txt'
    dd_loader = DailyDialogLoader(PATH_TO_DATA)
    dataloader = DataLoader(dd_loader, batch_size=16, shuffle=True, num_workers=0, collate_fn=PadCollate(pad_front=True))

    hidden_size = 256
    encoder1 = EncoderRNN(dd_loader.vocabulary.n_words, hidden_size).to(DEVICE)
    attn_decoder1 = DecoderRNN(hidden_size, dd_loader.vocabulary.n_words).to(DEVICE)

    generator = Generator(encoder1, attn_decoder1, criterion=nn.NLLLoss(ignore_index=0, size_average=False))

    trainIters(generator, dataloader, save_every=100)
