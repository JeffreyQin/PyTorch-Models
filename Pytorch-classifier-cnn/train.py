from tqdm import tqdm
import torch
from torch import nn 
from torch import optim

from cnn import CNN
from data import train_dataloader, test_dataloader


lr = 0.001
momentum = 0.9
epochs = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# train loop

for epoch in range(epochs):
    running_loss = 0.0
    for batch_index, (inputs, labels) in tqdm(enumerate(train_dataloader, 0)):

        prediction = model(inputs)
        loss = loss_fn(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'[{epoch + 1}] loss: {running_loss:.3f}')

# save model
PATH = './Pytorch-classifier-CNN/cifar_net.pth'
torch.save(model.state_dict(), PATH)
