# import packages
from collections import OrderedDict
import torch
from torch import nn

def get_training_model(in_features=4, hidden_dim=8, num_classes=3):
    # construct sequential neural network
    model = nn.Sequential(OrderedDict([
        ("hidden_layer_1", nn.Linear(in_features, hidden_dim)),
        ("activation_1", nn.ReLU()),
        ("output_layer", nn.Linear(hidden_dim, num_classes))
    ]));


    return model
