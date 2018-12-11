import torch
from torch.autograd import Variable
from math import ceil
from constants import *


def prepare_generator_batch(samples, start_letter=0, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """

    batch_size, seq_len = samples.size()

    inp = torch.zeros(batch_size, seq_len)
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len-1]

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target

def prepare_discriminator_data(pos_samples, neg_samples, gpu=False):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """

    true_samples = torch.nn.utils.rnn.pad_sequence([torch.cat((tensor, target),0) for (tensor, target) in pos_samples])
    neg_samples = neg_samples.squeeze(-1)
    false_target = torch.nn.utils.rnn.pad_sequence([target for (input, target) in pos_samples])

    neg_samples, false_target = neg_samples.to(DEVICE), false_target.to(DEVICE)

    false_samples = torch.cat((neg_samples, false_target),0).squeeze(1)

    true_samples, false_samples = true_samples.to(DEVICE), false_samples.to(DEVICE)

    inp = torch.cat((true_samples, false_samples), 0).type(torch.LongTensor).to(DEVICE)
    target = torch.ones(true_samples.size()[0] + false_samples.size()[0])
    target[true_samples.size()[0]:] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    inp = Variable(inp)
    target = Variable(target)

    inp = inp.to(DEVICE)
    target = target.to(DEVICE)

    return inp, target

def get_discriminator_inp_target(input_tensor, target_tensor, gen_tensor):
    gen_tensor = gen_tensor.squeeze(-1)

    true_samples = torch.cat((input_tensor, target_tensor), 1)
    false_samples = torch.cat((input_tensor, gen_tensor), 1)

    input_samples = [x for x in true_samples] + [x for x in false_samples]
    inp = torch.nn.utils.rnn.pad_sequence(input_samples).transpose(0,1)


    target = torch.ones(true_samples.size()[0] + false_samples.size()[0])
    target[true_samples.size()[0]:] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    inp = Variable(inp)
    target = Variable(target)

    inp = inp.to(DEVICE)
    target = target.to(DEVICE)

    return inp, target

def batchwise_sample(gen, num_samples, batch_size):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """

    samples = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]


def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, gpu=False):
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i+batch_size], start_letter, gpu)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll/(num_samples/batch_size)


def sample_true(dataloader, num_samples):
    lengths = [num_samples, len(dataloader.dataset)-num_samples]

    random_sample = torch.utils.data.random_split(dataloader.dataset, lengths)[0]
    return random_sample
