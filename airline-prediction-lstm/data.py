import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch


df = pd.read_csv('dataset.csv')
timeseries = df[["Passengers"]].values.astype('float32')

# dimension of timeseries is [144, 1] (NOT [144])

"""
# plot passenger timeseries
plt.plot(timeseries)
plt.show()
"""

train_set, test_set = train_test_split(timeseries, test_size=0.25, shuffle=False)

# function to group timeseries into windows of length "lookback", so that each input step into the lstm has "lookback" features
def create_dataset(dataset, lookback):
    x, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i: i+lookback]
        target = dataset[i+1: i+lookback+1]
        x.append(feature)
        y.append(target)
    return torch.tensor(np.array(x)), torch.tensor(np.array(y))

lookback=1
trainX, trainY = create_dataset(train_set, lookback=lookback)
testX, testY = create_dataset(test_set, lookback=lookback)

# dimension of trainX is [107, 1, 1] and trainY is [107, 1, 1]
