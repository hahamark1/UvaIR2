# -*- coding: utf-8 -*-
"""
Created on Dec 1 2018

@author: Jim
"""

import torch
import numpy as np
from models.model import AttnDecoderRNN, EncoderRNN, DecoderRNN, DecoderRNNwSoftmax, Generator
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


NUM_EPOCHS = 1000
PRINT_EVERY = 50
EVALUATE_EVERY = 300
SAVE_EVERY = 1000
TRAIN_GENERATOR_EVERY = 5

GEN_LEARNING_RATE = 0.001
DISC_LEARNING_RATE = 0.01



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, question=False):
    indexes = indexesFromSentence(lang, sentence)

    if question:
        indexes.append(EOU_token)
    else:
        indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)




def pretrain_generator(input_tensor, target_tensor, generator, gen_optimizer):
    gen_optimizer.zero_grad()

    # Generate a sentence and compute the loss based on the given target sentence
    gen_loss, generated_sentence = generator(input_tensor, target_tensor)

    # Train
    gen_loss.backward()
    gen_optimizer.step()

    return gen_loss


def pretrain_discriminator(input_tensor, target_tensor, generator, discriminator, disc_optimizer):
    disc_optimizer.zero_grad()
    
    # Generate a sentence to create an example for the discriminator
    _, generated_sentence = generator(input_tensor, target_tensor)
    
    # Determine the discriminator loss for both the true sample and the generated sample
    generated_disc_loss, generated_disc_out = discriminator(input_tensor, generated_sentence, true_sample=False)
    true_disc_loss, true_disc_out = discriminator(input_tensor, target_tensor, true_sample=True)

    # Compute the total discriminator loss
    disc_loss = generated_disc_loss + true_disc_loss

    # Train
    disc_loss.backward()
    disc_optimizer.step()

    return disc_loss


def train(input_tensor, target_tensor, generator, discriminator, adverserial_loss, gen_optimizer, disc_optimizer, train_generator):
    if train_generator:
        gen_optimizer.zero_grad()
    disc_optimizer.zero_grad()

    # Generate a sentence
    gen_loss, generated_sentence = generator(input_tensor, target_tensor)

    # Compute discriminator loss and output (0-1) for both the true and the generated sample
    generated_disc_loss, disc_generated = discriminator(input_tensor, generated_sentence, true_sample=False)
    true_disc_loss, disc_true = discriminator(input_tensor, target_tensor, true_sample=True)

    # Compute the total discriminator loss
    disc_loss = generated_disc_loss + true_disc_loss

    # Update the discriminator
    disc_loss.backward(retain_graph=train_generator)
    disc_optimizer.step()

    # Update the generator (if necessary) based on the discriminator rewards
    if train_generator:
        adv_loss = adverserial_loss(disc_generated, torch.ones(disc_generated.shape, device=DEVICE))
        adv_loss.backward()
        gen_optimizer.step()
        loss = adv_loss.item() + disc_loss.item()
    else:
        loss = disc_loss.item()
        adv_loss = 0
    
    return loss, (adv_loss, disc_loss)


def print_info(total_gen_loss, total_disc_loss, epoch, iteration):
    avg_gen_loss = total_gen_loss / PRINT_EVERY
    avg_disc_loss = total_disc_loss / PRINT_EVERY

    print('Epoch: {}, Iter: {}, Gen Loss: {:3.4f} Disc Loss: {:3.4f}'.format(
                            epoch, iteration, avg_gen_loss, avg_disc_loss))


def run_training(generator, discriminator, dataloader, gen_pre_train_epochs, disc_pre_train_epochs):

    gen_optimizer = optim.RMSprop(generator.parameters(), lr=GEN_LEARNING_RATE)
    disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=DISC_LEARNING_RATE)
    adverserial_loss = nn.BCELoss()

    total_gen_loss = 0
    total_disc_loss = 0

    for epoch in range(NUM_EPOCHS):

        pretrain_gen = epoch < gen_pre_train_epochs
        pretrain_disc = epoch < (gen_pre_train_epochs + disc_pre_train_epochs)

        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            iteration = epoch * len(dataloader) + i

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)

            if pretrain_gen:
                # Pre-train the generator
                gen_loss = pretrain_generator(input_tensor, target_tensor, generator, gen_optimizer)
                total_gen_loss += gen_loss

            elif pretrain_disc:
                # Pre-train the discriminator
                disc_loss = pretrain_discriminator(input_tensor, target_tensor, generator, discriminator, disc_optimizer)
                total_disc_loss += disc_loss

            else:
                # Do adversarial training
                # Train (n) discriminator steps for 1 generator step
                train_generator = i % TRAIN_GENERATOR_EVERY == 0
                _, all_losses = train(input_tensor, target_tensor, generator, discriminator, adverserial_loss,\
                                     gen_optimizer, disc_optimizer, train_generator)
                total_gen_loss += all_losses[0]
                total_disc_loss += all_losses[1]
                        
            # Print stuff and save the model every so often
            if iteration % PRINT_EVERY == 0 and iteration > 0:
                print_info(total_gen_loss, total_disc_loss, epoch, iteration)
                total_gen_loss, total_disc_loss = 0, 0

            if iteration % EVALUATE_EVERY == 0 and iteration > 0:
                # Take the first item of the batch to evaluate
                test_sentence, test_target_sentence = input_tensor[0, :], target_tensor[0, :]
                evaluate(generator, discriminator, test_sentence, test_target_sentence)

            if iteration % SAVE_EVERY == 0 and iteration > 0:
                torch.save(generator, os.path.join('saved_models', 'generator.pt'))
                torch.save(discriminator, os.path.join('saved_models', 'discriminator.pt'))



def evaluate(generator, discriminator, context_tensor, target_sentence):

    with torch.no_grad():
        context_tensor = context_tensor.view(1, -1)

        generated_sentence, generator_output = generator.generate_sentence(context_tensor)

        generator_output = torch.unsqueeze(generator_output, dim=0)
        target_sentence = torch.unsqueeze(target_sentence, dim=0)

        _, disc_out_gen = discriminator(context_tensor, generator_output, true_sample=False)
        _, disc_out_true = discriminator(context_tensor, target_sentence, true_sample=True)

        mean_disc_out_gen = torch.mean(disc_out_gen).item()
        mean_disc_out_true = torch.mean(disc_out_true).item()
        
        real_context = dataloader.dataset.vocabulary.list_to_sent(np.array(context_tensor[0]))
        real_reply = dataloader.dataset.vocabulary.list_to_sent(np.array(target_sentence[0]))
        generated_sentence = dataloader.dataset.vocabulary.list_to_sent(generated_sentence)

        print()
        print(real_context)
        print('>>')
        print(generated_sentence)
        print('==')
        print(real_reply)
        print('---------')
        print('Disc(0): {:3.2f} Disc(1): {:3.2f}'.format(mean_disc_out_gen, mean_disc_out_true))
        print('______________________________\n\n')


if __name__ == '__main__':

    PATH_TO_DATA =  'data/dailydialog/train/dialogues_train.txt'
    dd_loader = DailyDialogLoader(PATH_TO_DATA)
    dataloader = DataLoader(dd_loader, batch_size=16, shuffle=True, num_workers=0, collate_fn=PadCollate(pad_front=True))

    vocab_size = dd_loader.vocabulary.n_words
    hidden_size = 256

    gen_encoder = EncoderRNN(vocab_size, hidden_size).to(DEVICE)
    gen_decoder = DecoderRNNwSoftmax(hidden_size, vocab_size).to(DEVICE)
    generator = Generator(gen_encoder, gen_decoder, criterion=nn.NLLLoss(ignore_index=0))

    disc_encoder = EncoderRNN(vocab_size, hidden_size).to(DEVICE)
    disc_decoder = DecoderRNN(hidden_size, vocab_size).to(DEVICE)
    discriminator = Discriminator(disc_encoder, disc_decoder, hidden_size, vocab_size).to(DEVICE)

    gen_pre_train_epochs = 100
    disc_pre_train_epochs = 50

    run_training(generator, discriminator, dataloader, gen_pre_train_epochs, disc_pre_train_epochs)