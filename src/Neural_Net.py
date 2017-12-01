import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle


class BiLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, labelset_size, embeddings):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to label space
        self.linear = nn.Linear(hidden_dim, labelset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        inputs = [autograd.Variable(self.embeddings[word]) for word in sentence]
        inputs = torch.cat(inputs).view(len(inputs), 1, -1)

        out, self.hidden = self.lstm(
            inputs.view(len(sentence), 1, -1), self.hidden)

        # For each pair of hidden states, feed to linear layer
        # ASK ABOUT THIS- confused about when to do backprop, because
        # doesn't it only store the information about the most recent
        # linear operation? Do we do backprop after each pair?
        label_space = self.linear(out.view(len(sentence), -1))
        label_scores = F.log_softmax(label_space)
        return label_scores


# Make dict of labels
labels = pickle.load(open('data/2014/dm_edge_types.pickle', 'rb'))

# Read in GLOVE embeddings
embeddings = pickle.load(open('GLOVE_embeddings.pickle'))

EMBEDDING_DIM = 0 # Get this from GLOVE sizes
HIDDEN_DIM = 100
LABELSET_SIZE = len(labels)

model = BiLSTM(EMBEDDING_DIM, HIDDEN_DIM, LABELSET_SIZE, embeddings)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# For epochs
    # For each sentence
        # scores is output of model(sentence)

loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()