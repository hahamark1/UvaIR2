import torch
import torch.optim as optim
import torch.nn as nn
from models.AttnDecoderRNN import AttnDecoderRNN
from models.EncoderRNN import EncoderRNN
from models.DecoderRNN import DecoderRNN
from models.Generator import Generator
from constants import *
from dataloader.DailyDialogLoader import DailyDialogLoader, PadCollate
from torch.utils.data import DataLoader
import os
from utils.seq2seq_helper_funcs import plot_epoch_loss, plot_data
from nlgeval import NLGEval
from collections import defaultdict
nlgeval = NLGEval()

def load_model(length=MAX_UTTERENCE_LENGTH):
    """ Load the model if it is available"""
    return torch.load(os.path.join(PATH_TO_SAVE, 'generator_{}.pt'.format(length))).to(DEVICE)

def train(input_tensor, target_tensor, generator, optimizer):

    optimizer.zero_grad()
    loss, _ = generator(input_tensor, target_tensor)
    loss.backward()
    optimizer.step()
    target_length = target_tensor.shape[1]

    return loss.item() / target_length

def load_dataset(reversed=False):
    """ Load the training and test sets """


    train_dd_loader = DailyDialogLoader(PATH_TO_TRAIN_DATA, load=False)
    train_dataloader = DataLoader(train_dd_loader, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                            collate_fn=PadCollate())

    test_dd_loader = DailyDialogLoader(PATH_TO_TEST_DATA, load=True, reversed=reversed)
    test_dataloader = DataLoader(test_dd_loader, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=PadCollate())

    assert train_dd_loader.vocabulary.n_words == test_dd_loader.vocabulary.n_words

    return train_dd_loader, train_dataloader, test_dataloader


def trainIters(generator, train_dataloader, test_dataloader, num_epochs=EPOCHS, print_every=100,
               evaluate_every=500, save_every=500, learning_rate=0.01):

    optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3, threshold=0.5, min_lr=1e-4,
                                                     verbose=True)

    num_iters = len(train_dataloader)
    iter = 0
    iter_loss = 0
    losses = []
    metrics_dict = defaultdict(list)

    for epoch in range(num_epochs):

        epoch_loss = 0

        for i, (input_tensor, target_tensor) in enumerate(train_dataloader):

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)

            loss = train(input_tensor, target_tensor, generator, optimizer)

            iter_loss += loss
            epoch_loss += loss

            if iter % print_every == 0 and iter > 0:
                iter_loss_avg = iter_loss / print_every
                print('Average loss of the last {} iters {}'.format(print_every, iter_loss_avg))

                iter = 0
                iter_loss = 0

            if num_iters % evaluate_every == 0:
                test_sentence = input_tensor[0, :]
                test_target_sentence = target_tensor[0, :]

                real_test_sentence = train_dataloader.dataset.vocabulary.tokens_to_sent(test_sentence)
                real_target_sentence = train_dataloader.dataset.vocabulary.tokens_to_sent(test_target_sentence)

                decoded_words = evaluate(generator.encoder, generator.decoder, test_sentence)
                generated_sentence = train_dataloader.dataset.vocabulary.list_to_sent(decoded_words)

                print(real_test_sentence)
                print('>>')
                print(generated_sentence)
                print('==')
                print(real_target_sentence)
                print('-----------------------------')

            if num_iters % save_every == 0 and num_iters > 0:
                torch.save(generator,
                           os.path.join('saved_models', 'rnn_encoder_rnn_decoder_{}.pt'.format(MAX_UTTERENCE_LENGTH)))

            num_iters += 1
            iter += 1

        scheduler.step(epoch_loss)
        epoch_loss_avg = epoch_loss / i
        losses.append((epoch_loss_avg))
        plot_epoch_loss(losses)
        print('After epoch {} the loss is {}'.format(epoch, epoch_loss_avg))

        d = run_nlgeval(generator, test_dataloader)
        for key, value in d.items():
            metrics_dict[key].append(value)
            plot_data(metrics_dict[key], key)

        print('After epoch {} the metrics dict is: {}'.format(epoch, metrics_dict))

def evaluate(encoder, decoder, input_tensor, max_length=MAX_LENGTH):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        input_tensor = input_tensor.view(1, -1)
        input_length = input_tensor.shape[1]

        encoder_outputs = torch.zeros(max_length, 1, encoder.hidden_size, device=DEVICE)
        encoder_hidden = None

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei + max_length - input_length, :, :] = encoder_output[0, :, :]

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

    encoder.train()
    decoder.train()

    return decoded_words

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

        input_length = input_tensor.shape[1]

        encoder_outputs = torch.zeros(MAX_LENGTH, 1, encoder.hidden_size, device=DEVICE)
        encoder_hidden = None

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[:, ei], encoder_hidden)
            encoder_outputs[ei + MAX_LENGTH - input_length, :, :] = encoder_output[0, :, :]

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

        references.append([target_sent])
        hypothesis.append(corpus.list_to_sent(decoded_words))

    metrics_dict = nlgeval.compute_metrics(references, hypothesis)

    generator.train()

    return metrics_dict


if __name__ == '__main__':

    dd_loader, train_dataloader, test_dataloader = load_dataset()

    try:
        generator = load_model()
    except:
        encoder1 = EncoderRNN(dd_loader.vocabulary.n_words,
                              HIDDEN_SIZE,
                              num_layers=NUM_LAYERS,
                              LSTM='GRU')
        attn_decoder1 = AttnDecoderRNN(HIDDEN_SIZE,
                                       dd_loader.vocabulary.n_words,
                                       num_layers=NUM_LAYERS,
                                       LSTM='GRU')
        generator = Generator(encoder1,
                              attn_decoder1,
                              num_layers=NUM_LAYERS,
                              LSTM='GRU',
                              criterion=nn.CrossEntropyLoss(ignore_index=0, size_average=False)).to(DEVICE)

    print('Training the model with a max length of: {}'.format(MAX_UTTERENCE_LENGTH))

    trainIters(generator, train_dataloader, test_dataloader, save_every=50)
