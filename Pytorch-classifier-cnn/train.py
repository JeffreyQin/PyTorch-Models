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
    for batch_index, (inputs, labels) in enumerate(train_dataloader, 0):
        print('dadasdasasdadsadasd')
        prediction = model(inputs)
        loss = loss_fn(prediction, labels)
        print('dadasdasasdadsadasd')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('dsadas')
        running_loss += loss.item()
        if batch_index % 2000 == 1999:
            print(f'[{epoch + 1}, {sample_index + 1:5d}] loss: {running_loss / 2000:.3f}')