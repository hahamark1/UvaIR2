from __future__ import print_function
from math import ceil
import sys

from models.discriminator import *
import helpers
from dataloader.DailyDialogLoader import DailyDialogLoader, PadCollate
from models.generator import *
from torch.utils.data import DataLoader
from constants import *

from nlgeval import NLGEval

nlgeval = NLGEval()

CUDA = True if torch.cuda.is_available() else False
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 25
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 30
DISC_TRAIN_EPOCHS = 15
ADV_TRAIN_EPOCHS = 1000


PRINT_EVERY = 400
SAVE_EVERY = 1000
EVALUATE_EVERY = 100

PATH_TO_DATA = 'data/dailydialog/train/dialogues_train.txt'

def save_checkpoint(state, filename='checkpoint.m'):
    torch.save(state, filename)


def train_generator_MLE(generator, dataloader, gen_opt, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d : \n' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i, (input_tensor, target_tensor) in enumerate(dataloader):

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)

            gen_opt.zero_grad()

            loss, _, outputs = generator(input_tensor, target_tensor)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / dataloader.batch_size) % ceil(
                            ceil(len(dataloader) / float(dataloader.batch_size)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

            if i % SAVE_EVERY == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': generator.state_dict(),
                    'optimizer': gen_opt.state_dict(),
                }, filename='generator.m')

            if i % EVALUATE_EVERY == 0:
                test_sentence = input_tensor[0, :]
                test_target_sentence = target_tensor[0, :]
                generated_sentence = outputs[0, :]

                real_test_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_sentence)
                real_target_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_target_sentence)

                generated_sentence = dataloader.dataset.vocabulary.tokens_to_sent(generated_sentence)

                print(real_test_sentence)
                print('>>')
                print(generated_sentence)
                print('==')
                print(real_target_sentence)
                print('-----------------------------')



def train_generator_PG(generator, dataloader, gen_opt, dis):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for i, (input_tensor, target_tensor) in enumerate(dataloader):
        input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)


        output = generator.sample(input_tensor, length=target_tensor.shape[1])
        rewards = dis.batchClassify(target_tensor)

        gen_opt.zero_grad()
        pg_loss = generator.batchPGLoss(output, target_tensor, rewards)
        pg_loss.backward()
        gen_opt.step()

        if i % PRINT_EVERY == 0:
            print(' the PG_loss = {}'.format(pg_loss))

        if i % SAVE_EVERY == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': generator.state_dict(),
                'optimizer': gen_opt.state_dict(),
            }, filename='generator.m')

        if i % EVALUATE_EVERY == 0:
            test_sentence = input_tensor[0, :]
            test_target_sentence = target_tensor[0, :]
            generated_sentence = output[0, :]

            real_test_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_sentence)
            real_target_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_target_sentence)

            generated_sentence = dataloader.dataset.vocabulary.tokens_to_sent(generated_sentence)

            print(real_test_sentence)
            print('>>')
            print(generated_sentence)
            print('==')
            print(real_target_sentence)
            print('-----------------------------')


def train_discriminator(dataloader, discriminator, dis_opt, generator, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    pos_val = helpers.sample_true(dataloader, 100)
    neg_val = generator.sample(pos_val)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for epoch in range(epochs):
        print('epoch %d : \n' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0
        total_acc = 0

        for i, (input_tensor, target_tensor) in enumerate(dataloader):

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)
            gen_tensor = generator.sample(input_tensor)
            dis_inp, dis_target = helpers.get_discriminator_inp_target(input_tensor, target_tensor, gen_tensor.transpose(0,1))

            dis_opt.zero_grad()
            out = discriminator.batchClassify(dis_inp)
            loss_fn = nn.BCELoss()
            loss = loss_fn(out, dis_target)
            loss.backward()
            dis_opt.step()

            total_loss += loss.data.item()

            if i % PRINT_EVERY == 0:

                val_pred = discriminator.batchClassify(val_inp)
                print(' average_loss = %.8f' % (
                    total_loss))

            if i % SAVE_EVERY == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': discriminator.state_dict(),
                    'optimizer': dis_opt.state_dict(),
                }, filename='discriminator.m')

            if i % EVALUATE_EVERY == 0:
                test_sentence = input_tensor[0, :]
                test_target_sentence = target_tensor[0, :]
                generated_sentence = gen_tensor[0, :]

                real_test_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_sentence)
                real_target_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_target_sentence)

                generated_sentence = dataloader.dataset.vocabulary.tokens_to_sent(generated_sentence)

                print(real_test_sentence)
                print('>>')
                print(generated_sentence)
                print('==')
                print(real_target_sentence)
                print('-----------------------------')

def evaluate_test_set(generator, dataloader):

    references, hypothesis = [], []

    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(dataloader):

            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)
            output = generator.sample(input_tensor, length=target_tensor.shape[1])

            test_sentence = input_tensor[0, :]
            real_test_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_sentence)

            test_target_sentence = target_tensor[0, :]
            real_target_sentence = dataloader.dataset.vocabulary.tokens_to_sent(test_target_sentence)

            decoded_words = output[0, :]
            generated_sentence = dataloader.dataset.vocabulary.tokens_to_sent(decoded_words)

            if i % EVALUATE_EVERY == 0:
                print(real_test_sentence)
                print('>>')
                print(generated_sentence)
                print('==')
                print(real_target_sentence)
                print('-----------------------------')

            references.append([generated_sentence])
            hypothesis.append(real_target_sentence)

    metrics_dict = nlgeval.compute_metrics(references, hypothesis)

    return metrics_dict


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

# MAIN
if __name__ == '__main__':

    dd_loader, dataloader, test_dataloader = load_dataset()

    hidden_size = 256
    encoder1 = EncoderRNN(dd_loader.vocabulary.n_words, hidden_size).to(DEVICE)
    attn_decoder1 = DecoderRNN(hidden_size, dd_loader.vocabulary.n_words).to(DEVICE)

    generator = Generator(encoder1, attn_decoder1, hidden_size, hidden_size, dd_loader.vocabulary.n_words, criterion=nn.NLLLoss(ignore_index=0, size_average=False)).to(DEVICE)
    generator.load_state_dict(torch.load('generator.m')['state_dict'])

    dis = discriminator.Discriminator(hidden_size, hidden_size, dd_loader.vocabulary.n_words, MAX_SEQ_LEN, gpu=CUDA)
    dis.load_state_dict(torch.load('discriminator.m')['state_dict'])

    if CUDA:
        generator = generator.cuda()
        dis = dis.cuda()

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
    gen_optimizer.load_state_dict(torch.load('generator.m')['optimizer'])
    # train_generator_MLE(generator, dataloader, gen_optimizer, MLE_TRAIN_EPOCHS)

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    dis_optimizer.load_state_dict(torch.load('discriminator.m')['optimizer'])
    # train_discriminator(dataloader, dis, dis_optimizer, generator, DISC_TRAIN_EPOCHS)

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(generator, dataloader, gen_optimizer, dis)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dataloader, dis, dis_optimizer, generator, 1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': dis.state_dict(),
            'optimizer': dis_optimizer.state_dict(),
        }, filename='discriminator.m')

        metric = evaluate_test_set(generator, dataloader)
        with open("metrics.txt", "a") as myfile:
            myfile.write('\n\nEpoch: {}\n'.format(epoch))
            for key in metric.keys():
                myfile.write('{}: {}\n'.format(key, metric[key]))


        print(metric)
