import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import Linear, Module, ModuleList, Tanh
import torch.autograd as autograd
import torch.nn.functional as F
import cPickle as pickle

class MLP(Module):
    """ A multilayer perceptron with one hidden Tanh layer """
    def __init__(self,
                 input_dim,
                 hidden_layer_dim,
                 num_layers,
                 num_classes):
        super(MLP, self).__init__()

        self.num_layers = num_layers

        # create a list of layers of size num_layers
        layers = []
        for i in range(num_layers):
            d1 = input_dim if i == 0 else hidden_layer_dim
            d2 = hidden_layer_dim if i < (num_layers - 1) else num_classes
            layer = Linear(d1, d2)
            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(self, x):
        res = self.layers[0](x)
        for i in range(1, len(self.layers)):
            res = self.layers[i](Tanh(res))
        return res


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
            return []
        else:
            return self.embeddings[word]

    def forward(self, sentence):
        inputs = [autograd.Variable(torch.LongTensor(self.get_embedding(word))) for word in sentence]

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


def read_sentence(input):
    line = input.readline()
    lines = []

    if not line:
        return None, None
    else:
        while line and line.strip() is not "":
            lines.append(line)
            line = input.readline()

        tokens = []
        targets = []

        lines = lines[1:]
        num_tokens = len(lines)
        split_lines = []
        predicates = []

        for i in range(num_tokens):
            lines[i] = lines[i][:-1]
            split_line = lines[i].split('\t')
            tokens.append(split_line[1])
            split_lines.append(split_line)
            if split_line[5] is '+':
                predicates.append(i)

        for i in range(num_tokens):
            for j in range(num_tokens):
                labels_one_hot = np.zeros(LABELSET_SIZE)
                if j in predicates:
                    labels_one_hot[labels[split_lines[i][6 + predicates.index(j)]]] = 1
                else:
                    labels_one_hot[labels['_']] = 1
            targets.append(labels_one_hot)

        return tokens, targets


def get_embeddings(filename):
    input = open(filename, 'rb')
    embeddings = {}
    line = input.readline()
    while line:
        line = line[:-1]
        split_line = line.split(' ')
        embeddings[split_line[0]] = split_line[1:]
        line = input.readline()
    return embeddings

# Make dict of labels
labels = pickle.load(open('data/2015/dm_edge_types.pickle', 'rb'))

# Read in GLOVE embeddings
#embeddings = pickle.load(open('./data/glove_300.pickle', 'rb'))
embeddings = get_embeddings('./data/glove.6B.300d.txt')

EMBEDDING_DIM = 300
HIDDEN_DIM = 100
LABELSET_SIZE = len(labels)
EPOCHS = 300



model = BiLSTM(EMBEDDING_DIM, HIDDEN_DIM, LABELSET_SIZE, embeddings)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Try training on small batch of sentences to test if it is working
for i in range(EPOCHS):
    input = open('./data/2015/en.dm.sdp')
    input.readline()
    tokens, targets = read_sentence(input)
    while tokens and targets:
        # zero out optimizer
        optimizer.zero_grad()
        label_scores = model(tokens)
        loss = loss_function(label_scores, targets)
        print(loss)
        break
        # Add up the loss (make sure dimensions of sum are correct here)?
        # loss = loss.sum()
        loss.backward()
        optimizer.step()
        tokens, targets = read_sentence(input)

    # Test loss/accuracy on dev set

