# -*- coding: utf-8 -*-
"""
Created on Dec 1 2018

@author: Jim
"""

import torch
import numpy as np
from models.AttnDecoderRNN import AttnDecoderRNN
from models.EncoderRNN import EncoderRNN
from models.DecoderRNN import DecoderRNN
from models.Generator import Generator
from models.Discriminator import Discriminator
from evaluation.BlueEvaluator import BlueEvaluator
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
SAVE_EVERY = 300
TRAIN_GENERATOR_EVERY = 5

GEN_LEARNING_RATE = 0.001
DISC_LEARNING_RATE = 0.001

PATH_TO_TRAIN_DATA =  'data/dailydialog/train/dialogues_train.txt'
PATH_TO_TEST_DATA =  'data/dailydialog/test/dialogues_test.txt'


def pretrain_generator(input_tensor, target_tensor, generator, gen_optimizer):
    gen_optimizer.zero_grad()

    # Generate a sentence and compute the loss based on the given target sentence
    gen_loss, generated_sentence = generator(input_tensor, target_tensor)

    # Train
    gen_loss.backward()
    gen_optimizer.step()

    return gen_loss, generated_sentence


def pretrain_discriminator(input_tensor, target_tensor, generated_sentence, discriminator, disc_optimizer):
    disc_optimizer.zero_grad()
    
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
        total_gen_loss = gen_loss * adv_loss
        total_gen_loss.backward()
        gen_optimizer.step()
        loss = total_gen_loss.item() + disc_loss.item()
    else:
        loss = disc_loss.item()
        adv_loss = 0
    
    return loss, (adv_loss, disc_loss)


def print_info(total_gen_loss, total_disc_loss, epoch, iteration):
    avg_gen_loss = total_gen_loss / PRINT_EVERY
    avg_disc_loss = total_disc_loss / PRINT_EVERY

    print('Epoch: {}, Iter: {}, Gen Loss: {:3.4f} Disc Loss: {:3.4f}'.format(
                            epoch, iteration, avg_gen_loss, avg_disc_loss))


def run_training(generator, discriminator, dataloader, pre_train_epochs):

    gen_optimizer = optim.RMSprop(generator.parameters(), lr=GEN_LEARNING_RATE)
    disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=DISC_LEARNING_RATE)
    adverserial_loss = nn.BCELoss()

    scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=10, gamma=0.9)

    total_gen_loss = 0
    total_disc_loss = 0

    for epoch in range(NUM_EPOCHS):

        pretrain = epoch < pre_train_epochs
        # pretrain_disc = epoch < (gen_pre_train_epochs + disc_pre_train_epochs)

        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            iteration = epoch * len(dataloader) + i

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)

            if pretrain:
                # Pre-train the generator
                gen_loss, generated_sentence = pretrain_generator(input_tensor, target_tensor, generator, gen_optimizer)
                total_gen_loss += gen_loss

                # Pre-train the discriminator
                disc_loss = pretrain_discriminator(input_tensor, target_tensor, generated_sentence, discriminator, disc_optimizer)
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
                evaluate(generator, discriminator, test_sentence, test_target_sentence, dataloader)

            # if iteration % SAVE_EVERY == 0 and iteration > 0:

            #     torch.save({
            #         'epoch': epoch,
            #         'model': generator,
            #         'state_dict': generator.state_dict(),
            #         'optimizer' : gen_optimizer.state_dict(),
            #     }, os.path.join('saved_models', 'dp_gan_generator.pt'))

            #     torch.save({
            #         'epoch': epoch,
            #         'model': discriminator,
            #         'state_dict': discriminator.state_dict(),
            #         'optimizer' : disc_optimizer.state_dict(),
            #     }, os.path.join('saved_models', 'dp_gan_discriminator.pt'))



def evaluate(generator, discriminator, context_tensor, target_sentence, dataloader):

    with torch.no_grad():
        context_tensor = context_tensor.view(1, -1)

        # Generate a sentence given the context
        generated_sentence, generator_output = generator.generate_sentence(context_tensor)

        generator_output = torch.unsqueeze(generator_output, dim=0)
        target_sentence = torch.unsqueeze(target_sentence, dim=0)

        # Determine the discriminator outputs for both the true and the generated reply
        _, disc_out_gen = discriminator(context_tensor, generator_output, true_sample=False)
        _, disc_out_true = discriminator(context_tensor, target_sentence, true_sample=True)

        # Find the mean of these values (discriminator output is a probability per word in the reply)
        mean_disc_out_gen = torch.mean(disc_out_gen).item()
        mean_disc_out_true = torch.mean(disc_out_true).item()
        
        # Convert the tensors to readable sentences
        real_context = dataloader.dataset.vocabulary.list_to_sent(np.array(context_tensor[0]))
        real_reply = dataloader.dataset.vocabulary.list_to_sent(np.array(target_sentence[0]))
        generated_sentence = dataloader.dataset.vocabulary.list_to_sent(generated_sentence)

        # Print the results
        print()
        print('CONTEXT')
        print(real_context)
        print('GENERATED REPLY')
        print(generated_sentence)
        print('TRUE REPLY')
        print(real_reply)
        print('---------')
        print('Disc(0): {:3.2f} Disc(1): {:3.2f}'.format(mean_disc_out_gen, mean_disc_out_true))
        print('______________________________\n\n')


def evaluate_test_set(generator, train_dataloader, test_dataloader, max_length=MAX_LENGTH):

    encoder = generator.encoder
    decoder = generator.decoder

    BLUE = BlueEvaluator(train_dataloader.dataset.vocabulary.index2word)

    scores = []

    with torch.no_grad():

        for i, (input_tensor, target_tensor) in enumerate(test_dataloader):

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)

            batch_size = input_tensor.shape[0]
            input_length = input_tensor.shape[1]

            # print('batch size', batch_size)
            # print('input_length', input_length)

            encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size, device=DEVICE)
            encoder_hidden = None

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[:, ei], encoder_hidden)
                encoder_outputs[ei, :, :] = encoder_output[0, :, :]

            decoder_input = torch.tensor([[SOS_INDEX]], device=DEVICE).view(-1, 1)  # SOS
            # decoder_input = torch.tensor([[SOS_INDEX] * batch_size], device=DEVICE).transpose(0, 1)

            decoder_hidden = encoder_hidden

            decoded_words = []

            for di in range(MAX_WORDS_GEN):
                # print('dec in', decoder_input.shape)
                # print('dec hidden', decoder_hidden.shape)
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)

                if topi.item() == EOS_INDEX:
                    decoded_words.append(EOS_INDEX)
                    break
                else:
                    decoded_words.append(topi.item())

                decoder_input = topi.detach()

            score = BLUE.list_to_blue(decoded_words, target_tensor.cpu().tolist()[0])
            scores.append(score)

    average_score = sum(scores) / len(scores)
    return average_score



def load_dataset():
    """ Load the training and test sets """

    train_dd_loader = DailyDialogLoader(PATH_TO_TRAIN_DATA, load=False)
    train_dataloader = DataLoader(train_dd_loader, batch_size=16, shuffle=True, num_workers=0,
                            collate_fn=PadCollate())

    test_dd_loader = DailyDialogLoader(PATH_TO_TEST_DATA, load=True)
    test_dataloader = DataLoader(test_dd_loader, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=PadCollate())

    assert train_dd_loader.vocabulary.n_words == test_dd_loader.vocabulary.n_words

    return train_dd_loader, train_dataloader, test_dataloader


if __name__ == '__main__':

    dd_loader, train_dataloader, test_dataloader = load_dataset()

    vocab_size = dd_loader.vocabulary.n_words
    hidden_size = 256

    # Initialize the generator
    gen_encoder = EncoderRNN(vocab_size, hidden_size).to(DEVICE)
    gen_decoder = AttnDecoderRNN(hidden_size, vocab_size).to(DEVICE)
    generator = Generator(gen_encoder, gen_decoder, criterion=nn.CrossEntropyLoss(ignore_index=0, size_average=False))

    # Initialize the discriminator
    disc_encoder = EncoderRNN(vocab_size, hidden_size).to(DEVICE)
    disc_decoder = DecoderRNN(hidden_size, vocab_size).to(DEVICE)
    discriminator = Discriminator(disc_encoder, disc_decoder, hidden_size, vocab_size).to(DEVICE)

    # Number of epochs to pretrain the generator and discriminator, before performing adversarial training
    pre_train_epochs = 5
    run_training(generator, discriminator, train_dataloader, pre_train_epochs)

    # saved_gen = torch.load('saved_models/dp_gan_generator.pt')
    # saved_disc = torch.load('saved_models/dp_gan_discriminator.pt')
    # generator.load_state_dict(saved_gen['state_dict'])
    # discriminator.load_state_dict(saved_disc['state_dict'])

    # avg_score = evaluate_test_set(generator, train_dataloader, test_dataloader, max_length=MAX_LENGTH)
    # print('avg blue score:', avg_score)