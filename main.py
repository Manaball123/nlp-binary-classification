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
import tokenizer
import random
import copy




def train(model_to_train : model.RNN, input_tensor : torch.Tensor, target_tensor : torch.Tensor, learning_rate = 0.05, criterion = torch.nn.NLLLoss):
        
    device = utils.get_device()
    hidden = model_to_train.initHidden().to(device)

    model_to_train.zero_grad()

    for i in range(input_tensor.size()[0]):
        output, hidden = model_to_train(input_tensor[i], hidden)
    #it just is what it is, dont question it
    output_1d, target_1d = output.squeeze(1), target_tensor.squeeze((1,2)).type(torch.LongTensor).to(device)
    #god i hate this code
    loss = criterion(output_1d, target_1d)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model_to_train.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def main():
    session_id = str(time.time_ns())
    os.mkdir("./checkpoints/" + session_id)
    device = utils.get_device()
    device_cpu = torch.device("cpu")
    data_parser.get_files_index()

    rnn = model.RNN(config.TOKENS_N, 64, 128, 1)

    start_time = time.time()


    #print("Loading data")
    #train_data = data_parser.load_all_available()
    #print("Data loaded.")

    #moving to gpu
    rnn.to(device)

    criterion = torch.nn.NLLLoss()
    learning_rate = 0.2

    


    print_every = 1
    plot_every = 1



    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []


    start = time.time()

    train_entries = data_parser.get_available_entries()

    repeat_times = 4
    total_epochs = 5
    for epoch in range(total_epochs):
        current_entries = copy.deepcopy(train_entries)
        random.shuffle(current_entries)
        print("Executing epoch " + str(epoch))

        for it, entry in enumerate(current_entries):
            input_tensor = tokenizer.entry_id_to_tensor(entry).to(device)
            target_tensor = data_parser.get_target_tensor(entry).to(device)

            output, loss = train(rnn, input_tensor, target_tensor, learning_rate, criterion)
            current_loss += loss

            # Print ``iter`` number, loss, name and guess
            if it % print_every == 0:

                print(output)
                print("Iteration " + str(it) + "loss: " + str(loss))
           
            # Add current loss avg to list of losses
            if it % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0
            
            learning_rate = learning_rate ** 2
        torch.save(rnn.state_dict(), "./checkpoints/" + session_id + "/" + str(epoch) + ".pt")
        

    plt.figure()
    plt.plot(all_losses)
    plt.show()
    torch.save(rnn.state_dict(), "./model.pt")

    

if __name__ == "__main__":
    main()