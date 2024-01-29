import numpy as np
import torch
from torch import optim, utils, nn
from tqdm import tqdm

from lstm import LSTM
from data import train_dataloader, testX, testY

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTM().to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss() # MSE since regression task

epochs = 2000

# train loop

for epoch in range(epochs):
    model.train()

    for index, (batchX, batchY) in enumerate(train_dataloader):
        batchX, batchY = batchX.to(device), batchY.to(device)
        prediction = model(batchX)
        loss = loss_fn(prediction, batchY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            testX, testY = testX.to(device), testY.to(device)
            prediction = model(testX)
            loss = np.sqrt(loss_fn(prediction.cpu(), testY.cpu()))
            print(f'epoch: [{epoch}] loss this epoch: [{loss}]')





