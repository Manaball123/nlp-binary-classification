import torch
from torch import nn
import utils

class ClassifierModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassifierModel, self).__init__()
        #dumbest model ever
        self.input_to_output = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        return self.input_to_output(data)
    