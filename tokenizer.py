import numpy as np
from collections import Counter
import config
import torch
import utils

def word_to_idx(word : bytes):
    assert(len(word) == config.WORD_SIZE)
    return int.from_bytes(word, byteorder='little')

#credits: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
def bytes_to_tensor(bytes_in : bytes) -> torch.Tensor:
    
    out_words_len : int = len(bytes_in) // config.WORD_SIZE
    tensor = torch.zeros(out_words_len, 1, config.TOKENS_N)
    #prob gonna use a single byte as token anyway but adaptability etc etc

    for i in range(0, len(bytes_in), config.WORD_SIZE):
        tensor[i][0][word_to_idx(bytes_in[i:i + config.WORD_SIZE])] = 1.0
    return tensor

def entry_id_to_tensor(id : str) -> torch.Tensor:
    data_raw = utils.get_bin_data("dataset/samples/" + id)
    if not data_raw:
        return None
    #unfortunately this has way more performance overhead but it is what it is
    buf_tensor = torch.frombuffer(data_raw, offset=0, dtype=torch.uint8).long()
    return torch.nn.functional.one_hot(buf_tensor, 256).to(torch.float32)

