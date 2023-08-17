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
import random
import copy
#from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import pdb
from torch.utils.data import Dataset



def main():
    random.seed(42069)
    session_id = str(time.time_ns())
    os.mkdir("./checkpoints/" + session_id)
    device = utils.get_device()
    device_cpu = torch.device("cpu")
    data_parser.get_files_index()
    


    #freqs + entropy
    class_model = model.ClassifierModel(257,1).to(device)
    class_model.load_state_dict(torch.load("./model.pt"))
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
            

    #writer = SummaryWriter()
    
    failed_preds = 0
    successful_preds = 0
    total_preds = len(train_entries)
    #axis = plt.subplot(111)
    #line, = axis.plot(all_losses)
    current_entries = copy.deepcopy(train_entries)
    random.shuffle(current_entries)

    for it, entry in enumerate(current_entries):
        input_tensor = data_parser.load_entry(entry).to(device)
        target_tensor = data_parser.get_target_tensor(entry).to(device)
        output = class_model(input_tensor)
        if abs(target_tensor[0] - output[0]) > 0.5:
            failed_preds += 1
        else:
            successful_preds += 1
    print("Failed: ")
    print(failed_preds)
    print("Success: ")
    print(successful_preds)
    print("Success rate: " + str(successful_preds / total_preds))

    




    


if __name__ == "__main__":
    main()