import numpy as np
import torch
import model
import config
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import utils
import data_parser
import os



def main():
    files_available = []
    for file_path in os.listdir("dataset/samples"):
        files_available.append(file_path)
        


    device = utils.get_device()
    data_parser.get_files_index()

    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()

    batch_size = 50

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    rnn = model.RNN(config.TOKENS_N, 128, 1)


    #moving to gpu
    rnn.to(device)

    criterion = torch.nn.NLLLoss()
    learning_rate = 0.005

    def train(category_tensor, line_tensor):
        hidden = rnn.initHidden()

        rnn.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        return output, loss.item()
    

if __name__ == "__main__":
    main()