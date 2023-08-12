import pandas as pd
import numpy as np
import tokenizer
#DO NOT MODIFY DF
files_index : pd.DataFrame = None


def get_bin_data(filename : str) -> bytes:
    try:
        with open(filename, "rb") as f:
            data = f.read()
        return data
    except FileNotFoundError:
        print("[WARN]: filename " + filename + " not found!")
        return None

def get_files_index():
    global files_index
    if files_index == None:
        files_index = pd.read_csv("dataset/samples.csv", index_col="id", dtype={"id" : str})
        #only use "list" aka the category
        
        #float represents the confidence, 0 = def not malware, 1 = pretty much sure it is
        files_index["list"] = np.where(files_index["list"] == "Whitelist", 0.0, 1.0)
        files_index.rename(columns={"list" : "confidence"}, inplace=True)
        files_index = files_index[["confidence", "total", "positives"]]

        print(files_index.head(10))
        print(files_index.loc["143837"])
    return files_index


def load_entry(id : str) -> tuple:
    data_raw = get_bin_data("dataset/samples/" + id)
    if not data_raw:
        return None
    tensor = tokenizer.bytes_to_tensor(data_raw)
    
