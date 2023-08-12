import numpy as np
from collections import Counter
import config
import torch

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


