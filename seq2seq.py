import torch
from models.model import AttnDecoderRNN, EncoderRNN, Generator
import torch.optim as optim
import torch.nn as nn
from constants import *
from dataloader.DailyDialogLoader import DailyDialogLoader, PadCollate
from torch.utils.data import DataLoader
import os
from utils.seq2seq_helper_funcs import plot_blue_score, plot_epoch_loss
from evaluation.BlueEvaluator import BlueEvaluator
from nlgeval import NLGEval

nlgeval = NLGEval()

def load_model(length=MAX_UTTERENCE_LENGTH):
    """ Load the model if it is available"""

    return torch.load(os.path.join(PATH_TO_SAVE, 'generator_{}.pt'.format(length))).to(DEVICE)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

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

def load_dataset(reversed=False):
    """ Load the training and test sets """


    train_dd_loader = DailyDialogLoader(PATH_TO_TRAIN_DATA, load=False, reversed=reversed)
    train_dataloader = DataLoader(train_dd_loader, batch_size=16, shuffle=True, num_workers=0,
                            collate_fn=PadCollate())

    test_dd_loader = DailyDialogLoader(PATH_TO_TEST_DATA, load=True, reversed=reversed)
    test_dataloader = DataLoader(test_dd_loader, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=PadCollate())

    assert train_dd_loader.vocabulary.n_words == test_dd_loader.vocabulary.n_words

    return train_dd_loader, train_dataloader, test_dataloader


def trainIters(generator, train_dataloader, test_dataloader, LSTM, num_layers, model_name, num_epochs=3000, print_every=100,
               evaluate_every=100, save_every=100, learning_rate=0.001):

    optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    num_iters = len(train_dataloader)
    iter = 0
    iter_loss = 0
    blue_scores = []
    losses = []

    for epoch in range(num_epochs):

        epoch_loss = 0

        for i, (input_tensor, target_tensor) in enumerate(train_dataloader):

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)

            loss = train(input_tensor, target_tensor, generator, optimizer, generator.criterion)

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

            # if num_iters % save_every == 0 and num_iters > 0:

            num_iters += 1
            iter += 1

        scheduler.step()

        epoch_loss_avg = epoch_loss / i
        losses.append((epoch_loss_avg))
        print('After epoch {} the loss is {}'.format(epoch, epoch_loss_avg))
        average_score, metrics_dict = evaluate_test_set(generator, test_dataloader)
        blue_scores.append(average_score)
        plot_blue_score(blue_scores)
        plot_epoch_loss(losses)
        print('After epoch {} the average Blue score of the test set is: {}'.format(epoch, average_score))
        print(metrics_dict)
        torch.save(generator, os.path.join('saved_models', 'generator_{}_{}_{}_{}.pt'.format(MAX_UTTERENCE_LENGTH, LSTM, num_layers, model_name)))


def evaluate(encoder, decoder, input_tensor, max_length=MAX_LENGTH):
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

        return decoded_words

def evaluate_test_set(generator, test_dataloader, max_length=MAX_LENGTH):

    encoder = generator.encoder
    decoder = generator.decoder

    BLUE = BlueEvaluator(test_dataloader.dataset.vocabulary.index2word)


    scores = []
    metrics = []
    references, hypothesis = [], []

    print(len(test_dataloader))

    with torch.no_grad():

        for i, (input_tensor, target_tensor) in enumerate(test_dataloader):

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)

            batch_size = input_tensor.shape[0]
            input_length = input_tensor.shape[1]

            encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size, device=DEVICE)
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

            test_sentence = input_tensor[0, :]
            real_test_sentence = train_dataloader.dataset.vocabulary.tokens_to_sent(test_sentence)

            test_target_sentence = target_tensor[0, :]
            real_target_sentence = train_dataloader.dataset.vocabulary.tokens_to_sent(test_target_sentence)

            decoded_words = evaluate(generator.encoder, generator.decoder, test_sentence)
            generated_sentence = train_dataloader.dataset.vocabulary.list_to_sent(decoded_words)

            score = BLUE.list_to_blue(decoded_words, target_tensor.cpu().tolist()[0])

            references.append([generated_sentence])
            hypothesis.append(real_target_sentence)

            print(real_test_sentence)
            print('>>')
            print(generated_sentence)
            print('==')
            print(real_target_sentence)
            print('-----------------------------')
            scores.append(score)
    metrics_dict = nlgeval.compute_metrics(references, hypothesis)

    average_score = sum(scores) / len(scores)
    return average_score, metrics_dict


if __name__ == '__main__':

    dd_loader, train_dataloader, test_dataloader = load_dataset(reversed=True)
    model_name = 'seq2seq'

    # try:
    #     raise
    #     generator = load_model()
    # except:
    hidden_size = 256
    num_layers = 4
    LSTM = 'LSTM'
    encoder1 = EncoderRNN(dd_loader.vocabulary.n_words, hidden_size, num_layers=num_layers, LSTM=LSTM).to(DEVICE)
    attn_decoder1 = AttnDecoderRNN(hidden_size, dd_loader.vocabulary.n_words, num_layers=num_layers, LSTM=LSTM).to(DEVICE)
    generator = Generator(encoder1, attn_decoder1, num_layers=num_layers, criterion=nn.CrossEntropyLoss(ignore_index=0, size_average=False))

    print('The total number of parameters is: {}'.format(get_n_params(generator)))
    print('Training the model with a max length of: {}'.format(MAX_UTTERENCE_LENGTH))

    trainIters(generator, train_dataloader, test_dataloader, LSTM, num_layers, model_name, save_every=10000)
    # _,_ = evaluate_test_set(generator, test_dataloader)
