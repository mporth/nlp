import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from MLP import MLP


class BiLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, labelset_size, embeddings):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to label space
        self.linear = MLP(hidden_dim * 4, hidden_dim, 3, labelset_size)
        # ^ make sure hidden_dim * 4 is right here- bidirectional doubles it and so does the catting
        # when pairs are made
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def get_embedding(self, word):
        # deal with case of word
        word = word.lower()

        # deal with not found word
        if not self.embeddings[word]:
            return torch.zeros(1, 1, 300)
        else:
            return torch.FloatTensor(self.embeddings[word])

    def forward(self, sentence):
        inputs = [autograd.Variable(self.get_embedding(word)) for word in sentence]
        inputs = torch.cat(inputs).view(len(inputs), 1, -1)

        out, self.hidden = self.lstm(
            inputs.view(len(sentence), 1, -1), self.hidden)

        # For each pair of hidden states, feed to linear layer
        # do MLP on each pair, cat labels all together
        # in other file, loss is sum of loss of all of them
        # then backprop
        out_pairs = []
        for i in range(len(sentence)):
            for j in range(len(sentence)):
                pair = autograd.Variable(out[i].cat(out[j]))
                out_pairs.append(pair)

        out_pairs = autograd.Variable(out_pairs)

        label_space = self.linear(out.view(len(out_pairs), -1))
        # verify that label_space has the correct dimensions
        label_scores = F.log_softmax(label_space)
        return label_scores
