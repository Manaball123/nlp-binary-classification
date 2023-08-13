import torch
from torch import nn
import utils

class RNN(nn.Module):
    def __init__(self, input_size, embedding_out_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_out_size = embedding_out_size
        self.embedding = nn.Linear(input_size, embedding_out_size)

        #input 2 hidden
        self.i2h = nn.Linear(embedding_out_size + hidden_size, hidden_size)
        #hidden 2 output
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        embedding_pass = self.embedding(input)
        combined = torch.cat((embedding_pass, hidden), 0)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        #output = self.softmax(output)
        output = self.sigmoid(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.hidden_size)
    
