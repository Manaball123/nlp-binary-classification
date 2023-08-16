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
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import pdb


def is_nan(tensor : torch.Tensor) -> torch.Tensor:
    return tensor.isnan().any()




def train(model_to_train : torch.nn.Module, input_tensor : torch.Tensor, target_tensor : torch.Tensor, optimizer : torch.optim.Optimizer, learning_rate = 1e-3, criterion = torch.nn.BCELoss):
        
    device = utils.get_device()

    
    #hidden_backup = hidden.clone()

    output = model_to_train(input_tensor)
        #if(torch.isnan(hidden).any() or torch.isinf(hidden).any()):
        #    torch.set_printoptions(threshold = 10000)
            
        #hidden_backup = hidden.clone()
    #it just is what it is, dont question it
    #output_1d, target_1d = output.squeeze(1), target_tensor.squeeze((1,2)).type(torch.LongTensor).to(device)
    #god i hate this code

    model_to_train.zero_grad()
    optimizer.zero_grad()
    #only use the last output element
    loss = criterion(output[-1], target_tensor)
    torch.nn.utils.clip_grad_norm(model_to_train.parameters(), max_norm=5.0)
    loss.backward()
    optimizer.step()

        


    # Add parameters' gradients to their values, multiplied by learning rate
    

    return output, loss.item()


def main():
    random.seed(42069)
    session_id = str(time.time_ns())
    os.mkdir("./checkpoints/" + session_id)
    device = utils.get_device()
    device_cpu = torch.device("cpu")
    data_parser.get_files_index()

    #lstm = model.LSTM(config.TOKENS_N, 64, 2048, 1)
    transformer = model.TransformerModel(config.WORD_SIZE * 8, ).to(device)
    start_time = time.time()


    #print("Loading data")
    #train_data = data_parser.load_all_available()
    #print("Data loaded.")

    

    
    learning_rate = 1e-3
    benign_samples_n = 512
    malicious_samples_n = 512
    criterion = torch.nn.BCELoss()
    

    


    print_every = 1
    plot_every = 1



    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []


    start = time.time()

    all_entries = data_parser.get_available_entries()
    train_entries = []
    cur_benign = 0
    cur_malicious = 0
    for v in all_entries:
        if data_parser.get_entry_type(v):
            if cur_malicious >= malicious_samples_n:
                continue
            cur_malicious += 1
            
        else:
            if cur_benign >= benign_samples_n:
                continue
            cur_benign += 1
        train_entries.append(v)
            

    writer = SummaryWriter()
    
    total_epochs = 5
    #axis = plt.subplot(111)
    #line, = axis.plot(all_losses)
    for epoch in range(total_epochs):
        current_entries = copy.deepcopy(train_entries)
        random.shuffle(current_entries)
        print("\n\n\n------------------------------------Executing epoch " + str(epoch) + "------------------------------------\n\n\n")

        optimizer = torch.optim.Adam(transformer.parameters(), learning_rate)
        for it, entry in enumerate(current_entries):
            entry_start_time = time.time()
            utils.verbose_log("===========Processing " + entry + ", number " + str(it) + "============")
            input_tensor = tokenizer.entry_id_to_tensor(entry).to(device)
            target_tensor = data_parser.get_target_tensor(entry).to(device)
            utils.verbose_log("Data preprocessing complete. Target tensor: " + str(target_tensor) + ", Input tensor size: " + str(input_tensor.size()))

            output, loss = train(transformer, input_tensor, target_tensor, optimizer, learning_rate, criterion)
            current_loss += loss
            entry_end_time = time.time()

            utils.verbose_log("Procession complete. final output: " + str(output[-1]) + ", loss: " + str(loss))
            utils.verbose_log("Time taken: " + str(entry_end_time - entry_start_time) + " seconds.")


           
            # Add current loss avg to list of losses
            if it % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0
                #writer.add_scalar(tag= "hi", scalar_value=loss)
                

            
        learning_rate = learning_rate * 0.5
        torch.save(transformer.state_dict(), "./checkpoints/" + session_id + "/" + str(epoch) + ".pt")
        

    plt.show()
    torch.save(transformer.state_dict(), "./model.pt")

    

if __name__ == "__main__":
    main()