from __future__ import print_function
from math import ceil
import sys

import generator
import discriminator
import helpers
from dataloader.DailyDialogLoader import DailyDialogLoader, PadCollate
from models.model import *
from torch.utils.data import DataLoader
from constants import *

CUDA = True if torch.cuda.is_available() else False
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 25
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 50
ADV_TRAIN_EPOCHS = 100
ADV_TRAIN_STEPS = 1
POS_NEG_SAMPLES = 10000

PRINT_EVERY = 100
SAVE_EVERY = 1000
EVALUATE_EVERY = 100

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

oracle_samples_path = './oracle_samples.trc'
oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'

PATH_TO_DATA = 'data/dailydialog/train/dialogues_train.txt'


def save_checkpoint(state, filename='checkpoint.m'):
    torch.save(state, filename)

# def create_sentence(outputs):


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
        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
        #                                            start_letter=START_LETTER, gpu=CUDA)

        print(' average_train_NLL = %.4f' % (total_loss))


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
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': discriminator.state_dict(),
                'optimizer': dis_opt.state_dict(),
            }, filename='discriminator.m')


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


def train_discriminator(dataloader, discriminator, dis_opt, generator, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    pos_val = helpers.sample_true(dataloader, 100)
    neg_val = generator.sample(pos_val)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):

        for epoch in range(epochs):
            print('d-step %d epoch %d : \n' % (d_step + 1, epoch + 1), end='')
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
                total_acc += torch.sum((out > 0.5) == (dis_target > 0.5)).data.item()

                # if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                #         BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                #     print('.', end='')
                #     sys.stdout.flush()

                total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
                total_acc /= float(2 * POS_NEG_SAMPLES)

                if i % PRINT_EVERY == 0:

                    val_pred = discriminator.batchClassify(val_inp)
                    print(' average_loss = %.8f, train_acc = %.4f, val_acc = %.4f' % (
                        total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))

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

# MAIN
if __name__ == '__main__':

    # oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    # oracle.load_state_dict(torch.load(oracle_state_dict_path))
    # oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)

    dd_loader = DailyDialogLoader(PATH_TO_DATA)

    dataloader = DataLoader(dd_loader, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=PadCollate(pad_front=True))

    hidden_size = 256
    encoder1 = EncoderRNN(dd_loader.vocabulary.n_words, hidden_size).to(DEVICE)
    attn_decoder1 = DecoderRNN(hidden_size, dd_loader.vocabulary.n_words).to(DEVICE)

    generator = Generator(encoder1, attn_decoder1, GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, dd_loader.vocabulary.n_words, criterion=nn.NLLLoss(ignore_index=0, size_average=False))

    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()

    # gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, dd_loader.vocabulary.n_words, MAX_SEQ_LEN, gpu=CUDA)

    if CUDA:
        # oracle = oracle.cuda()
        generator = generator.cuda()
        dis = dis.cuda()
        # oracle_samples = oracle_samples.cuda()

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
    train_generator_MLE(generator, dataloader, gen_optimizer, MLE_TRAIN_EPOCHS)

    torch.save(generator.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dataloader, dis, dis_optimizer, generator, ADV_TRAIN_STEPS, ADV_TRAIN_EPOCHS)

    torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

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
        train_discriminator(dataloader, dis, dis_optimizer, generator, 1, 1)
