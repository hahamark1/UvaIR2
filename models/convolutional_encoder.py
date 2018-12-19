import torch.nn as nn
from models.grad_multiply import GradMultiply
import math
import torch
import torch.nn.functional as F
from constants import *


class FConvEncoder(nn.Module):
    """Convolutional encoder"""


    def __init__(self, vocab_size, embed_dim, convolutions=((HIDDEN_SIZE, 3),) * 20,
                 dropout=0.1, num_layers=1):

        super(FConvEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_layers = num_layers

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=PADDING_INDEX)
        self.embed_tokens.weight.data.normal_(mean=0, std=0.1)
        self.embed_tokens.weight[0].data.zero_()

        convolutions = extend_conv_spec(convolutions)
        self.hidden_size = convolutions[-1][0]

        in_channels = convolutions[0][0]

        self.fc1 = nn.Linear(embed_dim, in_channels)
        self.fc1.bias.data.zero_()
        self.dropout1 = nn.Dropout(p=dropout)

        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]

        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(nn.Linear(residual_dim, out_channels)
                                    if residual_dim != out_channels else None)

            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0

            self.convolutions.append(
                nn.Conv1d(in_channels, out_channels * 2, kernel_size, padding=padding)
            )

            self.batch_norms.append(nn.BatchNorm1d(out_channels * 2))

            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.fc2 = nn.Linear(in_channels, embed_dim)
        self.fc2.bias.data.zero_()

    def forward(self, src_tokens):

        # embed tokens and positions
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        residuals = [x]
        # temporal convolutions
        for proj, conv, norm, res_layer in zip(self.projections, self.convolutions, self.batch_norms, self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = F.dropout(x, p=self.dropout, training=self.training)
            if conv.kernel_size[0] % 2 == 1:

                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)

            # x = F.glu(x, dim=2)
            x = F.glu(x, dim=1)

            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

            x = norm(x)

        # T x B x C -> B x T x C
        # x = x.transpose(1, 0)

        # B x C x T -> B x T x C
        x = x.transpose(2, 1)

        # project back to size of embedding
        x = self.fc2(x)

        # scale gradients (this only affects backward, not forward)
        # Set self.num_attention_layers = 1 if line below needed
        # x = GradMultiply.apply(x, 1.0 / (2.0 * 1))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return y

def extend_conv_spec(convolutions):
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))
        else:
            raise Exception('invalid number of parameters in convolution spec ' + str(spec) + '. expected 2 or 3')
    return tuple(extended)

if __name__ == '__main__':

    batch_size = 16
    vocab_size = 399
    sequence_length = 40
    input_ = torch.empty(batch_size, sequence_length).random_(0, vocab_size).type(torch.LongTensor)
    ConvEncoder = FConvEncoder(vocab_size)
    print(input_.shape)

    output = ConvEncoder.forward(input_)
    print(output.shape)
