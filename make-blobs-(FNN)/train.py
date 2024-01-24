# import packages
from pyimagesearch import mlp
import torch
import sklearn
from torch.optim import SGD
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

# get next batch to train model
def next_batch(inputs, targets, batch_size):
    for i in range(0, inputs.shape[0], batch_size):
        yield (inputs[i: i + batch_size], targets[i: i + batch_size])


BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# get dataset
(X, y) = make_blobs(n_samples=1000, n_features=4, centers=3, cluster_std=2.5, random_state=95)
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.15, random_state=95)
# convert into pytorch tensors from numpy arrays
trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()

# instantiate neural network
mlp = mlp.get_training_model().to(DEVICE)
opt = SGD(mlp.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# training loop

train_log = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"

for epoch in range(0, EPOCHS):
    print("epoch: {}".format(epoch + 1))
    train_loss = 0
    train_acc = 0 # accuracy
    samples = 0
    mlp.train() 

    for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):
        (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
        
        # make prediction and calculate loss
        prediction = mlp(batchX)
        loss = loss_fn(prediction, batchY.long())

        loss.backward() # distribute gradient at weights
        opt.step() # update params
        opt.zero_grad() # clear gradient for next batch

        # update 
        train_loss += loss.item() * batchY.size(0)
        train_acc += (prediction.max(1)[1] == batchY).sum().item()
        samples += batchY.size(0)

        print(train_log.format(epoch + 1, (train_loss / samples), (train_acc / samples)))

# testing loop
test_loss = 0
test_acc = 0
samples = 0
mlp.eval() # disable dropout, etc.

with torch.no_grad(): # disable gradient
    for (batchX, batchY) in next_batch(testX, testY, BATCH_SIZE):
        (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))

        prediction = mlp(batchX)
        loss = loss_fn(prediction, batchY.long())

        test_loss += loss.item() * batchY.size(0)
        test_acc += (prediction.max(1)[1] == batchY).sum().item()
        samples += batchY.size(0)

    test_template = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
    print(test_template.format(epoch + 1, (test_loss / samples), (test_acc / samples)))
    print("")



