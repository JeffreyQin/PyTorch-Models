import torch
from torch import nn 

PATH = './joke-text-generator-lstm/lstm-model.pth'

# for more annotation, checkout airline-prediction-lstm
class LSTM(nn.Module):
    
    def __init__(self, dataset):
        super(LSTM, self).__init__()

        # note: num of lstm cells per layer depends on input sequence, not specified in param
        self.embedding_dim = 128
        self.hidden_dim = 128 # dim of hidden states
        self.num_layers = 3 # num of lstm layers
        self.n_vocab = len(dataset.unique_words) # num of words in dataset

        self.embedding_layer = nn.Embedding(
            num_embeddings=self.n_vocab, # total number of embeddings
            embedding_dim=self.embedding_dim # dimension of embedding vector
        )
        self.lstm_layer = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.2
        )
        self.linear_layer = nn.Linear(
            in_features=self.hidden_dim, # accepts output (not final hidden/cell states) from LSTM layer
            out_features=self.n_vocab # outputs a tensor indicating probability for all possible words
        )
    
    def forward(self, x, prev_state):
        embedded = self.embedding_layer(x)
        
        # recall that LSTM layer returns
        # 1. output (list of all hidden states): dimensions [batch_size, seq_length, hidden_dim]
        # 2. final hidden + cell states: dimensions [2, batch_size, hidden_dim]
        output, state = self.lstm_layer(embedded, prev_state)
        logits = self.linear_layer(output)
        return logits, state
    
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_dim),
                torch.zeros(self.num_layers, sequence_length, self.hidden_dim))