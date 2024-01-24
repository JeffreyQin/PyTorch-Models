from cnn import CNN
from preproc import trainX, trainY, valX, valY
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch


batch_size = 64
epochs = 25
lr = 0.07
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN().to(device)
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = CrossEntropyLoss()

trainY = trainY.long()
valY = valY.long()
trainX, valX = trainX.to(device), valX.to(device)
trainY, valY = trainY.to(device), valY.to(device)



def train():
    
    train_losses = []
    val_losses = []
    model.train()

    for epoch in range(epochs):
        train_prediction = model(trainX)
        val_prediction = model(valX)

        train_loss = loss_fn(train_prediction, trainY)
        val_loss = loss_fn(val_prediction, valY)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        print('Epoch: ', epoch + 1, '\t', 'loss : ', val_loss.item())

train()

