import torch
from models.convolutional_encoder import FConvEncoder
import torch.optim as optim
import torch.nn as nn
from constants import *
from dataloader.DailyDialogLoader import DailyDialogLoader, PadCollate
from torch.utils.data import DataLoader
import os
from utils.seq2seq_helper_funcs import plot_blue_score, plot_epoch_loss
from evaluation.BlueEvaluator import BlueEvaluator


def load_dataset():
    """ Load the training and test sets """


    train_dd_loader = DailyDialogLoader(PATH_TO_TRAIN_DATA, load=False)
    train_dataloader = DataLoader(train_dd_loader, batch_size=16, shuffle=True, num_workers=0,
                            collate_fn=PadCollate(pad_front=True))

    test_dd_loader = DailyDialogLoader(PATH_TO_TEST_DATA, load=True)
    test_dataloader = DataLoader(test_dd_loader, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=PadCollate(pad_front=True))

    assert train_dd_loader.vocabulary.n_words == test_dd_loader.vocabulary.n_words

    return train_dd_loader, train_dataloader, test_dataloader

def trainEpochs(encoder, train_dataloader, num_epochs=3000, print_every=100,
               evaluate_every=100, save_every=100, learning_rate=0.001):

    optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    num_iters = len(train_dataloader)
    iter = 0
    iter_loss = 0
    blue_scores = []
    losses = []

    for epoch in range(num_epochs):

        epoch_loss = 0

        for i, (input_tensor, target_tensor) in enumerate(train_dataloader):

            hidden, attention = encoder.forward(input_tensor)

            print(hidden.shape)


if __name__ == '__main__':

    dd_loader, train_dataloader, test_dataloader = load_dataset()

    FConvEncoder = FConvEncoder(dd_loader.vocabulary.n_words)

    trainEpochs(FConvEncoder, train_dataloader)

