import pandas as pd
import numpy as np
import torch
import os
import utils
#DO NOT MODIFY DF
files_index : pd.DataFrame = None




def get_files_index():
    global files_index
    if files_index == None:
        files_index = pd.read_csv("dataset/samples.csv", index_col="id", dtype={"id" : str})
        #only use "list" aka the category
        
        #float represents the confidence, 0 = def not malware, 1 = pretty much sure it is
        files_index["list"] = np.where(files_index["list"] == "Whitelist", 0.0, 1.0)
        files_index.rename(columns={"list" : "confidence"}, inplace=True)
        files_index = files_index[["confidence", "entropy", "total", "positives"]]

    return files_index



def load_entry(id : str) -> tuple:
    data = utils.get_json("dataset/output-samples/" + id + ".json")
    if not data:
        return None
    in_tensor = torch.zeros(1,1,257)
    for i in range(0,256):
        in_tensor[0][0][i] = data[i]
    in_tensor[0][0][256] = files_index.loc[id]["entropy"]
    out_tensor = torch.zeros(1,1,1)
    out_tensor[0][0][0] = files_index.loc[id]["confidence"]
    #should have marginally better performance
    #never occured to me that this, surprisingly, eats up vram
    #return (in_tensor,out_tensor)
    return in_tensor

def get_available_entries() -> list:
    files_available = []
    for file_path in os.listdir("dataset/output-samples"):
        files_available.append(os.path.splitext(file_path)[0])
    return files_available

def get_target_tensor(id : str) -> torch.Tensor:
    out_tensor = torch.zeros(1)
    out_tensor[0] = files_index.loc[id]["confidence"]
    return out_tensor



def get_entry_type(id : str) -> bool:
    conf = files_index.loc[id]["confidence"]
    if conf == 1.0:
        return True
    return False

    
