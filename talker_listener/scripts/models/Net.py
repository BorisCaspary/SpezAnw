#!/usr/bin/env python
##based on example given in task 3

from torch.nn import Linear, Module, ReLU, Sequential
import torch.nn.functional as F

class Net(Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()

        # split input according to your network's design,
        # in this case a single incoming image split into its height, width and channels
        # i.e. for original mnist (mono8) it is 28x28x1, for a rgb version it would be 28x28x3
        self.channels = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]

        # network layers
        # variant 1
        self.layer_1 = Sequential(
            Linear(in_features=self.height * self.width * self.channels, out_features=200),
            ReLU())
        self.layer_2 = Linear(in_features=200, out_features=200)  # variant 2
        self.layer_3 = Linear(in_features=200, out_features=10)

    def forward(self, x):
        x = x.view(-1, self.height * self.width)
        x = self.layer_1(x)  # variant 1
        x = F.relu(self.layer_2(x))  # variant 2
        x = self.layer_3(x)
        
        return F.log_softmax(x, dim=1)
