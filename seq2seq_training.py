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


def train(input_tensor, target_tensor, generator, discriminator, adverserial_loss, optimizer, disc_optimizer, criterion, train_generator, max_length=MAX_LENGTH):
    if train_generator:
        optimizer.zero_grad()
    disc_optimizer.zero_grad()
    # input_length = input_tensor.shape[1]

    gen_loss, generated_sentence = generator(input_tensor, target_tensor)
    generated_disc_loss, disc_generated = discriminator(input_tensor, generated_sentence, true_sample=False)
    true_disc_loss, disc_true = discriminator(input_tensor, target_tensor, true_sample=True)
    # print('disc generated', disc_generated, 'disc true', disc_true)
    
    if train_generator:
        adv_loss = adverserial_loss(disc_generated, torch.ones(disc_generated.shape, device=DEVICE))
    else: 
        adv_loss = 0
    disc_loss = generated_disc_loss + true_disc_loss

    # gen_loss.backward()
    if train_generator:
        adv_loss.backward(retain_graph=True)
        optimizer.step()
    
    disc_loss.backward()
    disc_optimizer.step()

    if train_generator:
        loss = adv_loss.item() + disc_loss.item()

    else:
        loss = disc_loss.item()


    return loss, (adv_loss, generated_disc_loss, true_disc_loss)


def trainIters(generator, discriminator, dataloader, num_epochs=300, print_every=1,
               plot_every=100, evaluate_every=100, save_every=100, train_generator_every=5,
               learning_rate=0.01):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    print_gen_loss_total = 0
    print_gen_disc_loss_total = 0
    print_true_disc_loss_total = 0
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.RMSprop(generator.parameters(), lr=0.001)
    disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
    adverserial_loss = nn.BCELoss()

    num_iters = len(dataloader)
    n_iters = 0

    for epoch in range(num_epochs):

        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            n_iters += 1

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)

            train_generator = i % train_generator_every == 0
            loss, all_losses = train(input_tensor, target_tensor, generator, discriminator, adverserial_loss, optimizer, disc_optimizer, criterion, train_generator)

            print_loss_total += loss
            plot_loss_total += loss
            print_gen_loss_total += all_losses[0]
            print_gen_disc_loss_total += all_losses[1]
            print_true_disc_loss_total += all_losses[2]

            if n_iters % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0

                print_gen_loss_avg = print_gen_loss_total / (print_every / train_generator_every)
                print_gen_loss_total = 0

                print_gen_disc_loss_avg = print_gen_disc_loss_total / print_every
                print_gen_disc_loss_total = 0

                print_true_disc_loss_avg = print_true_disc_loss_total / print_every
                print_true_disc_loss_total = 0

                print('Epoch: {}, Iter: {}, Loss: {:3.2f} Gen Loss: {:3.2f} Disc Loss: {:3.2f}'.format(
                                                                            epoch, i, print_loss_avg, print_gen_loss_avg, 
                                                                            print_gen_disc_loss_avg + print_true_disc_loss_avg))

            # if n_iters % plot_every == 0 and n_iters > 0:
            #     plot_loss_avg = plot_loss_total / plot_every
            #     plot_losses.append(plot_loss_avg)
            #     plot_loss_total = 0

            if n_iters % evaluate_every == 0:
                test_sentence = input_tensor[0, :]
                test_target_sentence = target_tensor[0, :]

                real_test_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_sentence)
                real_target_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_target_sentence)

                decoded_words, disc_losses = evaluate(generator.encoder, generator.decoder, discriminator, test_sentence, test_target_sentence)
                generated_sentence = dataloader.dataset.vocabulary.list_to_sent(decoded_words)
                print()
                print(real_test_sentence)
                print('>>')
                print(generated_sentence)
                print('==')
                print(real_target_sentence)
                print('---------')
                print('Disc(0): {} Disc(1): {}'.format(disc_losses[0], disc_losses[1]))
                print('______________________________\n\n')

            if n_iters % save_every == 0 and n_iters > 0:
                torch.save(generator, os.path.join('saved_models', 'generator.pt'))

        plot_loss_avg = plot_loss_total / num_iters
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
        print('\nAfter epoch {} the loss is {}\n'.format(epoch, plot_loss_avg))

    showPlot(plot_losses)

def evaluate(encoder, decoder, discriminator, input_tensor, target_sentence, max_length=20):
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
        generator_output = torch.zeros(MAX_WORDS_GEN, device=DEVICE).long()

        for di in range(MAX_WORDS_GEN):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            topv, topi = decoder_output.data.topk(1)
            generator_output[di] = topi.view(-1)

            if topi.item() == EOS_INDEX:
                decoded_words.append(EOS_INDEX)
                generator_output = generator_output[:di]
                break
            else:
                decoded_words.append(topi.item())
            
            decoder_input = topi.detach()
        
        # If it hasn't been predicted, add an EOS_TOKEN to the end of the generated response
        # if generator_output[-1].item() != EOS_INDEX:
        #     decoded_words.append(EOS_INDEX)
        #     generator_output = torch.cat((generator_output, torch.tensor([EOS_INDEX], device=DEVICE)))
        
        # Remove padding tokens from the target sentence
        # target_sentence = target_sentence[target_sentence.nonzero()].squeeze(dim=1)
        
        generator_output = torch.unsqueeze(generator_output, dim=0)
        target_sentence = torch.unsqueeze(target_sentence, dim=0)

        _, disc_out_gen = discriminator(input_tensor, generator_output, true_sample=False)
        _, disc_out_true = discriminator(input_tensor, target_sentence, true_sample=True)

        return decoded_words, (disc_out_gen.item(), disc_out_true.item())

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
    discriminator = Discriminator(encoder=disc_encoder, decoder=disc_decoder, hidden_size=hidden_size, vocab_size=dd_loader.vocabulary.n_words).to(DEVICE)

    trainIters(generator, discriminator, dataloader, print_every=100, save_every=100)
