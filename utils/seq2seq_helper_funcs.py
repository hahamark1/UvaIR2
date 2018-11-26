from __future__ import unicode_literals, print_function, division
import os
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
from constants import *

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def plot_blue_score(scores):
    """ Plot the blue scores, given a list of scores"""

    plt.figure()
    plt.plot(scores)
    plt.title('BLUE score over epochs', fontsize=17)
    plt.xlabel('Epochs', fontsize=17)
    plt.ylabel('BLUE score', fontsize=17)

    plt.savefig(os.path.join('figures', 'blue_score_{}.png'.format(MAX_LENGTH)))
    plt.close()

def plot_epoch_loss(losses):
    """ Plot the losses of each epoch """

    plt.figure()
    plt.plot(losses)
    plt.title('Losses over epochs', fontsize=17)
    plt.xlabel('Epochs', fontsize=17)
    plt.ylabel('Loss', fontsize=17)

    plt.savefig(os.path.join('figures', 'losses_{}.png'.format(MAX_LENGTH)))
    plt.close()



