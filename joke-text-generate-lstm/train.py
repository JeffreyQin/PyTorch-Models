import argparse
import torch
import numpy as np 
from torch import nn, optim, utils
from lstm import LSTM, PATH
from dataset import Dataset

def train(dataset, model, args):
    model.train()

    dataloader = utils.data.DataLoader(dataset, batch_size=args.batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        # get hidden/cell states of LSTM from previous epoch
        h_state, c_state = model.init_state(args.sequence_length)

        for index, (batchX, batchY) in enumerate(dataloader):
            # (h_state, c_state) passed as parameters set initial hidden/cell states to final hidden/cell states from last batch
            # new (h_state, c_state) produced are the final hidden/cell states from this epoch
            prediction, (h_state, c_state) = model(batchX, (h_state, c_state))

            # _____PRINT DEBUG ON TRANSPOSE DIM
            loss = loss_fn(prediction.transpose(1,2), batchY)

            # detach initial hidden/cell states so BPTT for current batch won't affect previous batches
            h_state = h_state.detach()
            c_state = c_state.detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch + 1} batch {index} loss: {loss.item()}')


# setup args
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--sequence-length', type=int, default=4)
parser.add_argument('--learning-rate', type=float, default=0.001)
args = parser.parse_args()

dataset = Dataset(args)
model = LSTM(dataset)

train(dataset, model, args)

torch.save(model.state_dict(), PATH)