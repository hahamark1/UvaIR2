import torch
import nltk

class BlueEvaluator(objec):
    """
    The BLUE score evaluator
    """


    def __init__(self, index2word):
        self.index2word = index2word


    def batch_to_blue(self, batch, target):
        """
        Inputs is a torch tensor of shape [batch x seq_len] with indices
        """
