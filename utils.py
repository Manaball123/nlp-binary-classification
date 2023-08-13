import torch

def get_device() -> torch.device:
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        #print("GPU is available")
        return torch.device("cuda")
    
    print("GPU not available, CPU used")
    return torch.device("cpu")

def get_bin_data(filename : str) -> bytes:
    try:
        with open(filename, "rb") as f:
            data = f.read()
        return data
    except FileNotFoundError:
        print("[WARN]: filename " + filename + " not found!")
        return None
        