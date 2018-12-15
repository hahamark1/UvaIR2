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
from models.convolutional_encoder import FConvEncoder
from models.convolutional_generator import ConvEncoderRNNDecoder
from models.Generator import Generator
from models.Discriminator import Discriminator
from models.ConvolutionalDiscriminator import ConvDiscriminator
from evaluation.BlueEvaluator import BlueEvaluator
import random
import time
from utils.seq2seq_helper_funcs import showPlot, asMinutes, timeSince, plot_data
import torch.optim as optim
import torch.nn as nn
from constants import *
from dataloader.DailyDialogLoader import DailyDialogLoader, PadCollate
from torch.utils.data import Dataset, DataLoader
import os
from nlgeval import NLGEval
from collections import defaultdict
nlgeval = NLGEval()


NUM_EPOCHS = 70
PRINT_EVERY = 50
EVALUATE_EVERY = 300
SAVE_EVERY = 300
TRAIN_GENERATOR_EVERY = 5

GEN_LEARNING_RATE = 0.01
DISC_LEARNING_RATE = 0.01

PATH_TO_TRAIN_DATA =  'data/dailydialog/train/dialogues_train.txt'
PATH_TO_TEST_DATA =  'data/dailydialog/test/dialogues_test.txt'


def pretrain_generator(input_tensor, target_tensor, generator, gen_optimizer):
    gen_optimizer.zero_grad()

    # Generate a sentence and compute the loss based on the given target sentence
    gen_loss, generated_sentence = generator(input_tensor, target_tensor)

    # Train
    gen_loss.backward()
    gen_optimizer.step()

    return gen_loss, generated_sentence.detach()


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
    generated_disc_loss, disc_generated = discriminator(input_tensor, generated_sentence.detach(), true_sample=False)
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


def run_training(start_epoch, generator, discriminator, train_dataloader, test_dataloader, pre_train_epochs, convolutional, gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler):
    
    adverserial_loss = nn.BCELoss()
    

    total_gen_loss = 0
    total_disc_loss = 0

    metrics_dict = defaultdict(list)

    for epoch in range(NUM_EPOCHS + pre_train_epochs):
        if epoch < start_epoch:
            continue

        pretrain = epoch < pre_train_epochs

        for i, (input_tensor, target_tensor) in enumerate(train_dataloader):
            iteration = epoch * len(train_dataloader) + i

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

            if iteration % EVALUATE_EVERY == 0 and iteration > 0:
                # Take the first item of the batch to evaluate
                test_sentence, test_target_sentence = input_tensor[0, :], target_tensor[0, :]
                evaluate(generator, discriminator, test_sentence, test_target_sentence, test_dataloader)

            if iteration % SAVE_EVERY == 0 and iteration > 0:

                torch.save({
                    'epoch': epoch,
                    'model': generator,
                    'state_dict': generator.state_dict(),
                    'optimizer' : gen_optimizer.state_dict(),
                }, os.path.join('saved_models', 'dp_gan_generator_{}2.pt'.format('convolutional' if convolutional else 'recurrent')))

                torch.save({
                    'epoch': epoch,
                    'model': discriminator,
                    'state_dict': discriminator.state_dict(),
                    'optimizer' : disc_optimizer.state_dict(),
                }, os.path.join('saved_models', 'dp_gan_discriminator_{}2.pt'.format('convolutional' if convolutional else 'recurrent')))

        d = run_nlgeval(generator, test_dataloader)
        for key, value in d.items():
            metrics_dict[key].append(value)
            plot_data(metrics_dict[key], key)

        gen_scheduler.step(total_gen_loss)
        disc_scheduler.step(total_disc_loss)
        total_gen_loss, total_disc_loss = 0, 0


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

def run_nlgeval(generator, test_dataloader):

    generator.eval()

    references = []
    hypothesis = []

    corpus = test_dataloader.dataset.vocabulary

    encoder = generator.encoder
    decoder = generator.decoder

    for i, (input_tensor, target_tensor) in enumerate(test_dataloader):

        input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)
        target_sent = corpus.tokens_to_sent(target_tensor.view(-1))

        batch_size = input_tensor.shape[0]
        input_length = input_tensor.shape[1]

        target_length = target_tensor.shape[1]
        encoder_hidden = encoder.initHidden(batch_size=batch_size, num_layers=NUM_LAYERS)

        encoder_outputs = torch.zeros(MAX_LENGTH, batch_size, encoder.hidden_size, device=DEVICE)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei + MAX_LENGTH - input_length, :, :] = encoder_output[0, :, :]


        decoder_input = torch.tensor([[SOS_INDEX]], device=DEVICE).transpose(0, 1)
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

        references.append([target_sent])
        hypothesis.append(corpus.list_to_sent(decoded_words))

    metrics_dict = nlgeval.compute_metrics(references, hypothesis)

    generator.train()

    return metrics_dict

def load_dataset():
    """ Load the training and test sets """

    train_dd_loader = DailyDialogLoader(PATH_TO_TRAIN_DATA, load=False)
    train_dataloader = DataLoader(train_dd_loader, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                            collate_fn=PadCollate())

    test_dd_loader = DailyDialogLoader(PATH_TO_TEST_DATA, load=True)
    test_dataloader = DataLoader(test_dd_loader, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=PadCollate())

    assert train_dd_loader.vocabulary.n_words == test_dd_loader.vocabulary.n_words

    return train_dd_loader, train_dataloader, test_dataloader


if __name__ == '__main__':

    convolutional = False

    dd_loader, train_dataloader, test_dataloader = load_dataset()

    vocab_size = dd_loader.vocabulary.n_words

    if convolutional:
        ConvEncoder = FConvEncoder(vocab_size, HIDDEN_SIZE)
        AttnDecoderRNN = AttnDecoderRNN(HIDDEN_SIZE, vocab_size, num_layers=NUM_LAYERS, LSTM='GRU')
        generator = ConvEncoderRNNDecoder(ConvEncoder, AttnDecoderRNN, num_layers=NUM_LAYERS,
                                     criterion=nn.CrossEntropyLoss(ignore_index=0, size_average=False), dpgan=True).to(DEVICE)

        # Initialize the discriminator
        disc_encoder = FConvEncoder(vocab_size, HIDDEN_SIZE)
        disc_decoder = DecoderRNN(HIDDEN_SIZE, vocab_size, num_layers=NUM_LAYERS, LSTM='GRU')
        discriminator = ConvDiscriminator(disc_encoder, disc_decoder, vocab_size, num_layers=NUM_LAYERS).to(DEVICE)

    else:
        # Initialize the generator
        gen_encoder = EncoderRNN(vocab_size, HIDDEN_SIZE, num_layers=NUM_LAYERS, LSTM='GRU')
        gen_decoder = AttnDecoderRNN(HIDDEN_SIZE, vocab_size, num_layers=NUM_LAYERS, LSTM='GRU')
        generator = Generator(gen_encoder, gen_decoder, criterion=nn.CrossEntropyLoss(ignore_index=0, size_average=False)).to(DEVICE)

        # Initialize the discriminator
        disc_encoder = EncoderRNN(vocab_size, HIDDEN_SIZE, num_layers=NUM_LAYERS, LSTM='GRU')
        disc_decoder = DecoderRNN(HIDDEN_SIZE, vocab_size, num_layers=NUM_LAYERS, LSTM='GRU')
        discriminator = Discriminator(disc_encoder, disc_decoder, HIDDEN_SIZE, vocab_size, NUM_LAYERS).to(DEVICE)

    gen_optimizer = optim.RMSprop(generator.parameters(), lr=GEN_LEARNING_RATE)
    disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=DISC_LEARNING_RATE)

    gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, factor=0.2, patience=3, threshold=0.5, min_lr=1e-4,
                                                     verbose=True)
    disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(disc_optimizer, factor=0.2, patience=3, threshold=0.5, min_lr=1e-4,
                                                     verbose=True)

    saved_gen = torch.load('saved_models/dp_gan_generator_recurrent1.pt')
    saved_disc = torch.load('saved_models/dp_gan_discriminator_recurrent1.pt')
    generator.load_state_dict(saved_gen['state_dict'])
    discriminator.load_state_dict(saved_disc['state_dict'])

    gen_optimizer.load_state_dict(saved_gen['optimizer'])
    disc_optimizer.load_state_dict(saved_disc['optimizer'])

    epoch = saved_gen['epoch']



    # Number of epochs to pretrain the generator and discriminator, before performing adversarial training
    pre_train_epochs = 5
    run_training(epoch, generator, discriminator, train_dataloader, test_dataloader, pre_train_epochs, convolutional, gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler)

    

    # avg_score = evaluate_test_set(generator, train_dataloader, test_dataloader, max_length=MAX_LENGTH)
    # print('avg blue score:', avg_score)