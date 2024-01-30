import torch
from torch import utils

import pandas as pd 
from collections import Counter

class Dataset(utils.data.Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.args = args

        # list of all words & list of unique words
        self.words = self.load_words()
        self.unique_words = self.get_unique_words()

        # dictionaries that assign indices to words
        self.index_to_word = {index: word for (index, word) in enumerate(self.unique_words)}
        self.word_to_index = {word: index for (index, word) in enumerate(self.unique_words)}

        self.words_indices = [self.word_to_index[w] for w in self.words]
    
    # return list of all words
    def load_words(self):
        train_df = pd.read_csv('data.csv')
        text_concatenated = train_df['Joke'].str.cat(sep=' ')
        return text_concatenated.split(' ')
    
    # return list of unique words in the order of decreasing frequency
    def get_unique_words(self):
        word_ctr = Counter(self.words)
        return sorted(word_ctr, key=word_ctr.get, reverse=True) 

    # length of dataset (num of contiguous sentences of length sequence_length)
    def __len__(self):
        return len(self.words_indices) - self.args.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indices[index : index + self.args.sequence_length]),
            torch.tensor(self.words_indices[index + 1 : index + self.args.sequence_length + 1])
        )