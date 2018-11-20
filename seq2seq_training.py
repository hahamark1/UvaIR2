# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:56:42 2018

@author: Joris
"""

import torch
from models.model import AttnDecoderRNN, EncoderRNN, DecoderRNN, Generator
from models.Discriminator import Discriminator
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


def train(input_tensor, target_tensor, generator, discriminator, optimizer, criterion, max_length=MAX_LENGTH):

    optimizer.zero_grad()
    input_length = input_tensor.shape[1]
    loss, generated_sentence = generator(input_tensor, target_tensor)
    generated_disc_loss = discriminator(input_tensor, generated_sentence, true_sample=False)
    true_disc_loss = discriminator(input_tensor, target_tensor, true_sample=True)

    print('generated_disc_loss:', generated_disc_loss)
    print('true_disc_loss:', true_disc_loss)
    
    # TODO: add discriminator loss as well
    loss.backward()
    optimizer.step()
    target_length = target_tensor.shape[1]

    return loss.item() / target_length


def trainIters(generator, discriminator, dataloader, num_epochs=30, print_every=100,
               plot_every=100, evaluate_every=100, save_every=100,
               learning_rate=0.01):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0, size_average=False)

    num_iters = len(dataloader)
    n_iters = 0

    for epoch in range(num_epochs):

        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            n_iters += 1

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)

            loss = train(input_tensor, target_tensor, generator, discriminator, optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss

            if n_iters % print_every == 0 and n_iters > 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('{} Epoch:[{}/{}] Iter:[{}/{}] Loss: {}'.format(timeSince(start, float(n_iters) / n_iters),
                                                                      epoch, num_epochs, i, len(dataloader), print_loss_avg))

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

        plot_loss_avg = plot_loss_total / num_iters
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
        print('After epoch {} the loss is {}'.format(epoch, plot_loss_avg))

    showPlot(plot_losses)

def evaluate(encoder, decoder, input_tensor, max_length=20):
    with torch.no_grad():

        input_length = max(input_tensor.size())

        input_tensor = input_tensor.view(1, -1)

        encoder_outputs = torch.zeros(max_length, 1, encoder.hidden_size, device=DEVICE)
        encoder_hidden = None

        for ei in range(min(input_length, max_length)):
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

        # print(decoded_words)
        return decoded_words

if __name__ == '__main__':

    PATH_TO_DATA =  'data/dailydialog/train/dialogues_train.txt'
    dd_loader = DailyDialogLoader(PATH_TO_DATA)
    dataloader = DataLoader(dd_loader, batch_size=16, shuffle=True, num_workers=0, collate_fn=PadCollate(pad_front=True))

    hidden_size = 256
    encoder1 = EncoderRNN(dd_loader.vocabulary.n_words, hidden_size).to(DEVICE)
    attn_decoder1 = DecoderRNN(hidden_size, dd_loader.vocabulary.n_words).to(DEVICE)
    generator = Generator(encoder1, attn_decoder1, criterion=nn.NLLLoss(ignore_index=0, size_average=False))

    disc_encoder = EncoderRNN(dd_loader.vocabulary.n_words, hidden_size).to(DEVICE)
    disc_decoder = DecoderRNN(hidden_size, dd_loader.vocabulary.n_words).to(DEVICE)
    discriminator = Discriminator(encoder=disc_encoder, decoder=disc_decoder)

    trainIters(generator, discriminator, dataloader, print_every=100, save_every=100)
