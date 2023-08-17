import torch
from torch import nn
import utils

class RNN(nn.Module):
    def __init__(self, input_size, embedding_out_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_out_size = embedding_out_size
        self.embedding = nn.Linear(input_size, embedding_out_size)

        #input 2 hidden
        self.i2h = nn.Linear(embedding_out_size + hidden_size, hidden_size)
        #hidden 2 output
        self.h2o = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        ) 
 
        self.hidden = torch.zeros(hidden_size).to(utils.get_device())

    def forward(self, input):
        embedding_pass = self.embedding(input)
        combined = torch.cat((embedding_pass, self.hidden), 0)
        self.hidden = self.i2h(combined)
        output = self.h2o(self.hidden)

        return output

    def init_states(self):
        self.hidden = torch.zeros(self.hidden_size).to(utils.get_device())
        


class LSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_t = None
        self.hidden_size = hidden_size
        #AKA long term memory
        self.c_t = None
        self.c_size = hidden_size 
        self.device = utils.get_device()
        #concated embedding + hidden
        self.intermediate_size = embedding_size + hidden_size
        self.embedding_network = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.Softmax(1)
        
        )

        self.forget_gate_network = nn.Sequential(
            nn.Linear(self.intermediate_size, self.c_size),
            nn.Sigmoid()
        )
        self.information_gate_network = nn.Sequential(
            nn.Linear(self.intermediate_size, self.c_size),
            nn.Sigmoid()
        )
        #generates the matrix that is multiplied by information
        self.memory_signal_network = nn.Sequential(
            nn.Linear(self.intermediate_size, self.c_size),
            nn.Tanh()
        )
        #multiplied by memory(C)
        self.hidden_output_gate_network = nn.Sequential(
            nn.Linear(self.intermediate_size, self.hidden_size),
            nn.Sigmoid()
        )
        #gets an output(finally) from the transformed hidden
        self.hidden_to_output_network = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        self.memory_transformer = nn.Sequential(
            nn.Tanh()
        )
      


    def init_states(self):
        #hidden
        self.hidden_t = torch.zeros(self.hidden_size).to(self.device)
        #memory cells
        self.c_t = torch.zeros(self.c_size).to(self.device)

    #data should be a onehot vector
    def forward(self, data):
        self.init_states()
        #torch.set_printoptions(threshold = 10000)
        #print(data.size())
        embedding = self.embedding_network(data)
        #print(embedding.size())
        for i in range(data.size()[0]):
            intermediates = torch.cat((embedding[i], self.hidden_t), 0)

            forget_t = self.forget_gate_network(intermediates)
            

            info_t = self.information_gate_network(intermediates)
            signal_t = self.memory_signal_network(intermediates)

            info_gated_t = info_t * signal_t

            #forget some info
            self.c_t = forget_t * self.c_t
            #add new info
            self.c_t = torch.add(self.c_t, info_gated_t)

            #transform memory for hidden
            mem_transformed = self.memory_transformer(self.c_t)

            #
            out_gate_t = self.hidden_output_gate_network(intermediates)
            #also selectively forget some stuff
            self.hidden_t = mem_transformed * out_gate_t

        output = self.hidden_to_output_network(self.hidden_t)
        return output


class ClassifierModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassifierModel, self).__init__()
        #dumbest model ever
        self.input_to_output = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        return self.input_to_output(data)
    