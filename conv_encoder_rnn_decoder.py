import torch
from models.convolutional_encoder import FConvEncoder
import torch.optim as optim
import torch.nn as nn
from models.AttnDecoderRNN import AttnDecoderRNN
from models.convolutional_generator import ConvEncoderRNNDecoder
from constants import *
from dataloader.DailyDialogLoader import DailyDialogLoader, PadCollate
from torch.utils.data import DataLoader
import os
from utils.seq2seq_helper_funcs import plot_blue_score, plot_epoch_loss
from evaluation.BlueEvaluator import BlueEvaluator
from nlgeval import NLGEval
nlgeval = NLGEval()

def load_model():
    return torch.load(os.path.join('saved_models', 'conv_encoder_rnn_decoder_{}.pt'.format(MAX_UTTERENCE_LENGTH)))

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


def train(input_tensor, target_tensor, generator, optimizer):

    optimizer.zero_grad()
    loss = generator(input_tensor, target_tensor)
    loss.backward()
    nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.1)
    optimizer.step()
    target_length = target_tensor.shape[1]

    return loss.item() / target_length


def trainIters(generator, train_dataloader, test_dataloader, num_epochs=3000, print_every=1000,
               evaluate_every=1000, save_every=1000, learning_rate=0.25):

    optimizer = optim.SGD(generator.parameters(), lr=learning_rate, momentum=0.99, nesterov=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=10, threshold=0.5, min_lr=1e-4, verbose=True)

    num_iters = len(train_dataloader)
    iter = 0
    iter_loss = 0
    blue_scores = []
    losses = []

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
                torch.save(generator, os.path.join('saved_models', 'conv_encoder_rnn_decoder_{}.pt'.format(MAX_UTTERENCE_LENGTH)))

            num_iters += 1
            iter += 1

        scheduler.step(epoch_loss)
        epoch_loss_avg = epoch_loss / i
        losses.append((epoch_loss_avg))
        print('After epoch {} the loss is {}'.format(epoch, epoch_loss_avg))
        average_score = evaluate_test_set(generator, test_dataloader)
        blue_scores.append(average_score)
        plot_blue_score(blue_scores)
        plot_epoch_loss(losses)
        print('After epoch {} the average Blue score of the test set is: {}'.format(epoch, average_score))

        metrics_dict = run_nlgeval(CERD, test_dataloader)
        print('After epoch {} the metrics dict is: {}'.format(epoch, metrics_dict))


def evaluate(encoder, decoder, input_tensor, max_length=MAX_LENGTH):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        input_tensor = input_tensor.view(1, -1)
        input_length = input_tensor.shape[1]

        encoder_outputs = torch.zeros(max_length, 1, encoder.hidden_size, device=DEVICE)
        encoder_hidden = encoder.forward(input_tensor).transpose(0, 1)

        for ei in range(input_length):
            encoder_outputs[ei + max_length - input_length, :, :] = encoder_hidden[ei, :, :]

        decoder_input = torch.tensor([[SOS_INDEX]], device=DEVICE).transpose(0, 1)
        decoder_hidden = encoder_hidden[-1, :, :].unsqueeze(0)

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

def evaluate_test_set(generator, test_dataloader, max_length=MAX_LENGTH):

    encoder = generator.encoder
    decoder = generator.decoder

    BLUE = BlueEvaluator(test_dataloader.dataset.vocabulary.index2word)

    scores = []

    with torch.no_grad():

        for i, (input_tensor, target_tensor) in enumerate(test_dataloader):

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)

            batch_size = input_tensor.shape[0]
            input_length = input_tensor.shape[1]

            encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size, device=DEVICE)
            encoder_hidden = encoder.forward(input_tensor).transpose(0, 1)

            for ei in range(input_length):
                encoder_outputs[ei + (max_length - input_length), :, :] = encoder_hidden[ei, :, :]

            decoder_input = torch.tensor([[SOS_INDEX]], device=DEVICE).transpose(0, 1)
            decoder_hidden = encoder_hidden[-1, :, :].unsqueeze(0)

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

            score = BLUE.list_to_blue(decoded_words, target_tensor.cpu().tolist()[0])
            scores.append(score)

    average_score = sum(scores) / len(scores)
    return average_score

def run_nlgeval(generator, test_dataloader):


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

        encoder_outputs = torch.zeros(MAX_LENGTH, batch_size, encoder.hidden_size, device=DEVICE)
        encoder_hidden = encoder.forward(input_tensor).transpose(0, 1)

        for ei in range(input_length):
            encoder_outputs[ei + (MAX_LENGTH - input_length), :, :] = encoder_hidden[ei, :, :]

        decoder_input = torch.tensor([[SOS_INDEX]], device=DEVICE).transpose(0, 1)
        decoder_hidden = encoder_hidden[-1, :, :].unsqueeze(0)

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
    return metrics_dict


if __name__ == '__main__':

    dd_loader, train_dataloader, test_dataloader = load_dataset()

    embed_dim = 512

    try:
        CERD = load_model()
        print('Succesfully loaded the model')
    except:
        ConvEncoder = FConvEncoder(dd_loader.vocabulary.n_words, embed_dim=embed_dim)
        AttnDecoderRNN = AttnDecoderRNN(hidden_size=embed_dim, output_size=dd_loader.vocabulary.n_words)
        CERD = ConvEncoderRNNDecoder(ConvEncoder, AttnDecoderRNN, criterion=nn.CrossEntropyLoss(ignore_index=0, size_average=False)).to(DEVICE)

    trainIters(CERD, train_dataloader, test_dataloader, num_epochs=3000, save_every=10000000)
    run_nlgeval(CERD, test_dataloader)

