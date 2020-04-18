import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT = "word" # unit of tokenization (char, word, sent)
TASK = None # task (None, word-segmentation, sentence-segmentation)
RNN_TYPE = "LSTM" # LSTM or GRU
NUM_DIRS = 2 # unidirectional: 1, bidirectional: 2
NUM_LAYERS = 2
BATCH_SIZE = 10
HRE = (UNIT == "sent") # hierarchical recurrent encoding
EMBED = {"lookup": 100} # embeddings (char-cnn, char-rnn, lookup, sae)
EMBED_SIZE = sum(EMBED.values())
HIDDEN_SIZE = 100
DROPOUT = 0.5
LEARNING_RATE = 2e-4
EVAL_EVERY = 10
SAVE_EVERY = 10

PAD = "<PAD>" # padding
SOS = "<SOS>" # start of sequence
EOS = "<EOS>" # end of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

CUDA = torch.cuda.is_available()
#CUDA = False
torch.manual_seed(0) # for reproducibility
# torch.cuda.set_device(0)

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor
randn = lambda *x: torch.randn(*x).cuda() if CUDA else torch.randn
zeros = lambda *x: torch.zeros(*x).cuda() if CUDA else torch.zeros

KEEP_IDX = False # use the existing indices when adding more training data
NUM_DIGITS = 4 # number of digits to print
