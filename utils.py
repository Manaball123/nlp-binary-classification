import torch

def get_device() -> torch.device:
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("GPU is available")
        return torch.device("cuda")
    
    print("GPU not available, CPU used")
    return torch.device("cpu")
        