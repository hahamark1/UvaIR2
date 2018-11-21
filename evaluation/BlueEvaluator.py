from nltk.translate.bleu_score import sentence_bleu
from constants import *

class BlueEvaluator(object):
    """
    The BLUE score evaluator
    """


    def __init__(self, index2word):
        self.index2word = index2word
        self.padding_token = PADDING_TOKEN


    def list_to_blue(self, generated, target):
        """ Function that takes to lists with indices, and returns the blue score """

        generated_sentence = [self.index2word[token] for token in generated]
        generated_sentence = [word for word in generated_sentence if word != self.padding_token]

        target_sentence = [self.index2word[token] for token in target]
        target_sentence = [word for word in target_sentence if word != self.padding_token]

        blue_score = sentence_bleu([target_sentence], generated_sentence, weights=(1./3, 1./3, 1./3, 0))

        return blue_score


