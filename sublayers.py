import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, heads=1):
        super(AttentionLayer, self).__init__()
        self.w_query = nn.Conv1d(embedding_dim, heads*embedding_dim, 1)
        self.w_key = nn.Conv1d(embedding_dim, heads*embedding_dim, 1)
        self.w_value = nn.Conv1d(embedding_dim, heads*embedding_dim, 1)
        nn.init.normal_(self.w_query.weight, mean=0, std=np.sqrt(
            2.0 / (embedding_dim + (heads*embedding_dim))))
        nn.init.normal_(self.w_key.weight, mean=0, std=np.sqrt(
            2.0 / (embedding_dim + (heads*embedding_dim))))
        nn.init.normal_(self.w_value.weight, mean=0, std=np.sqrt(
            2.0 / (embedding_dim + (heads*embedding_dim))))
        self.scale_factor = 1. / np.sqrt(embedding_dim)
        self.w_out = nn.Linear(heads*embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input, mask=None):
        if mask is None:
            masked_input = input
        else:
            masked_input = mask * input
        q = self.w_query(masked_input.transpose(1, 2))
        k = self.w_key(masked_input.transpose(1, 2))
        v = self.w_value(masked_input.transpose(1, 2))
        print(torch.bmm(q, k.transpose(1, 2)).shape)
        attention_weights = F.softmax(
            torch.bmm(q, k.transpose(1, 2)), dim=-1) / self.scale_factor
        output = torch.bmm(attention_weights, v).transpose(1, 2)
        output = self.w_out(output)
        output = self.layer_norm(input + output)
        return output, attention_weights


class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(PositionwiseFeedForwardLayer, self).__init__()
        self.w_1 = nn.Conv1d(embedding_dim, hidden_dim, 1)
        self.w_2 = nn.Conv1d(hidden_dim, embedding_dim, 1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        output = self.w_2(F.relu(self.w_1(input.transpose(1, 2))))
        output = self.layer_norm(input + output.transpose(1, 2))
        return output



