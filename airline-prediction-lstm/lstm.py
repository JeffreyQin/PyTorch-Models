import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__():

        # input_size = number of features in each input time step
        # hidden_size = dimension of hidden state vectors passed in between consecutive lstm cells
        # num_layers = number of lstm cells in the same layer
        # batch_first=true indicates that batch_size is the first dimension of the input
        self.lstm_layer = nn.LSTM(input_size=4, hidden_size=50, num_layers=1, batch_first=True)

        # generates a single output (float) based on all hidden states given to the output layer (all at once)
        self.linear_layer = nn.Linear(50, 1)

    def forward(self, x):
        # x is the output of the lstm layer, with dimensions (batch_size, sequence_length, num_features)
        # _ is the final hidden + cell state from the lstm layer
        x, _ = self.lstm_layer(x) 
        x = self.linear_layer(x)
        return x