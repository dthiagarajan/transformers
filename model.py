''' Websites that are useful:
http://blog.varunajayasiri.com/ml/transformer.html
http://jalammar.github.io/illustrated-transformer/
'''

from sublayers import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


''' Taken from http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding'''


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, embedding_dim):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(seq_len, embedding_dim)
        position = torch.arange(0., seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., embedding_dim, 2) *
                             -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, seq_len, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(seq_len, embedding_dim)

    def forward(self, input):
        return self.pos_encoding(self.embedding(input) * np.sqrt(embedding_dim))


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, heads=1):
        super(EncoderLayer, self).__init__()
        self.attention = AttentionLayer(embedding_dim, heads=heads)
        self.pwff = PositionwiseFeedForwardLayer(embedding_dim, hidden_dim)

    def forward(self, input, detailed=False):
        output, attention_weights = self.attention(input)
        output = self.pwff(output)
        if detailed:
            return output, attention_weights
        else:
            return output


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, heads=1):
        super(DecoderLayer, self).__init__()
        self.attention = AttentionLayer(embedding_dim, heads=heads)
        self.pwff = PositionwiseFeedForwardLayer(embedding_dim, hidden_dim)

    def forward(self, input, detailed=False):
        output, attention_weights = self.attention(input)
        output = self.pwff(output)
        if detailed:
            return output, attention_weights
        else:
            return output
