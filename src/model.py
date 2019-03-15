from torch import nn
from torch.nn import functional as F
import torch
import pdb
import sys
from os.path import dirname
sys.path.append('..')
from config import SIZE, SHAPE

class simplerNN(nn.Module):
    def __init__(self, shape=SHAPE, input_channels=2, channels_out=8, layer_count=5):
        super(simplerNN, self).__init__()

        self.shape = shape
        layers = []
        for _ in range(layer_count):
            layers.append(nn.Conv2d(input_channels, channels_out, kernel_size = 3,padding=1, bias=True))
            input_channels = channels_out
            layers.append(nn.BatchNorm2d(input_channels))
            layers.append(nn.ReLU())

        self.front = nn.Sequential(*layers)

        self.prob = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size = 3, padding=1, bias=True),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(),
                nn.Conv2d(input_channels, input_channels, kernel_size = 3, padding=1, bias=True),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(),
                nn.Conv2d(input_channels, 2, kernel_size = 1),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                nn.Conv2d(2, 1, kernel_size = 1),
                nn.BatchNorm2d(1),
                nn.ReLU()
                )

        self.softmax = nn.Softmax(dim=1)

        self.value = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size = 3, bias=True),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(),
                nn.Conv2d(input_channels, input_channels, kernel_size = 3, bias=True),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(),
                nn.Conv2d(input_channels, 1, 1),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.AvgPool2d(shape),
                nn.Tanh()
                )

    def forward(self, boards):
        # computing forward pass
        front = self.front(boards)
        prob  = self.prob(front)
        prob  = self.softmax(prob.view(prob.shape[0],-1))
        value = self.value(front)

        return value, prob
