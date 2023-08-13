import numpy as np
import torch
import model
import config
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import utils
import data_parser
import os
import math
import time
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker




def main():
    


    session_id = str(time.time_ns())
    os.mkdir("./checkpoints/" + session_id)
    device = utils.get_device()
    device_cpu = torch.device("cpu")
    data_parser.get_files_index()

    rnn = model.RNN(config.TOKENS_N, 64, 128, 1)

    train_data = data_parser.load_all_available()

    #moving to gpu
    rnn.to(device)

    criterion = torch.nn.NLLLoss()
    learning_rate = 0.2

    def train(input_tensor : torch.Tensor, target_tensor : torch.Tensor):
        
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        hidden = rnn.initHidden().to(device)

        rnn.zero_grad()

        for i in range(input_tensor.size()[0]):
            output, hidden = rnn(input_tensor[i], hidden)
        #it just is what it is, dont question it
        output_1d, target_1d = output.squeeze(1), target_tensor.squeeze((1,2)).type(torch.LongTensor).to(device)
        #god i hate this code
        loss = criterion(output_1d, target_1d)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        return output, loss.item()


    print_every = 1
    plot_every = 1



    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    repeat_times = 4
    for iter in range(len(train_data) * repeat_times):
        
        output, loss = train(*train_data[iter % len(train_data)])
        current_loss += loss

        # Print ``iter`` number, loss, name and guess
        if iter % print_every == 0:

            print(output)
            print("Iteration loss: " + str(loss))
           

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
        torch.save(rnn.state_dict(), "./checkpoints/" + session_id + "/" + str(iter) + ".pt")
        learning_rate = learning_rate ** 2
    

    plt.figure()
    plt.plot(all_losses)
    plt.show()
    torch.save(rnn.state_dict(), "./model.pt")

    

if __name__ == "__main__":
    main()