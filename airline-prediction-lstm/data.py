import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch
from torch import utils


df = pd.read_csv('dataset.csv')
timeseries = df[["Passengers"]].values.astype('float32')

# dimension of timeseries is [144, 1] (NOT [144])

"""
# plot passenger timeseries
plt.plot(timeseries)
plt.show()
"""

train_set, test_set = train_test_split(timeseries, test_size=0.25, shuffle=False)

# function to group timeseries into windows of length "lookback"
def create_dataset(dataset, lookback):
    x, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i: i+lookback]
        target = dataset[i+1: i+lookback+1]
        x.append(feature)
        y.append(target)
    return torch.tensor(np.array(x)), torch.tensor(np.array(y))

lookback=4
trainX, trainY = create_dataset(train_set, lookback=lookback)
testX, testY = create_dataset(test_set, lookback=lookback)

# dimension of trainX is [108 - lookback, lookback, 1] and trainY is [108 - lookback, lookback, 1]

train_dataloader = utils.data.DataLoader(utils.data.TensorDataset(trainX, trainY), batch_size = 8, shuffle=True)
test_dataloader = utils.data.DataLoader(utils.data.TensorDataset(testX,testY), batch_size=8, shuffle=False)