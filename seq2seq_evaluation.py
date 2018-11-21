import numpy as np
import torch
from constants import *
from dataloader.DailyDialogLoader import DailyDialogLoader, PadCollate
from torch.utils.data import Dataset, DataLoader
from evaluation.BlueEvaluator import BlueEvaluator

def load_model(length=MAX_UTTERENCE_LENGTH):
    """ Load the model if it is available"""

    return torch.load(os.path.join(PATH_TO_SAVE, 'generator_{}.pt'.format(length))).to(DEVICE)

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

            score = BLUE.list_to_blue(decoded_words, target_tensor.cpu().tolist()[0])
            scores.append(score)

    average_score = sum(scores) / len(scores)
    return average_score


if __name__ == '__main__':

    PATH_TO_DATA =  'data/dailydialog/test/dialogues_test.txt'
    test_dd_loader = DailyDialogLoader(PATH_TO_DATA)
    test_dataloader = DataLoader(test_dd_loader, batch_size=1, shuffle=True, num_workers=0, collate_fn=PadCollate(pad_front=True))

    hidden_size = 256

    generator = load_model()

    evaluate_test_set(generator, test_dataloader)

