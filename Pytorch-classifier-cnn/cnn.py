import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extraction_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10) 
        )
    
    def forward(self, x):
        x = self.feature_extraction_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x

