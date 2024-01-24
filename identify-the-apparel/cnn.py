import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d
from torch.optim import Adam, SGD


# implement model

class CNN(Module):

    # init
    def __init__(self):
        super(CNN, self).__init__()

        self.feature_extraction_layers = Sequential(
            
            # in_feature_map_num, out_feature_map_num
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            # in_feature_map_num
            BatchNorm2d(4),
            ReLU(inplace=True), # inplace means input will be modifed directly for output
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            # in_features, out_features
            Linear(4 * 7 * 7, 10)
        )
    
    # prediction
    def forward(self, x):
        x = self.feature_extraction_layers(x)
        x = x.view(x.size(0), -1) # flatten. dimension: [# samples / batch, 1]
        x = self.linear_layers(x)
        return x