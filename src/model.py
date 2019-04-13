from torch import nn
from torch.nn import functional as F
import torch
import pdb
import sys
from config import *


class simplerNN(nn.Module):
    def __init__(self):
        super(simplerNN, self).__init__()

        self.front = self.conv_layers(CHANNELS, CONV_CHANNELS, FRONT_LAYER_CNT)

        self.policy_conv = self.conv_layers(CONV_CHANNELS, CONV_CHANNELS, POLICY_LAYER_CNT)
        self.policy = nn.Sequential(
                nn.Conv2d(CONV_CHANNELS, 1, kernel_size=3, padding=1),
                )

        self.value_conv = self.conv_layers(CONV_CHANNELS, CONV_CHANNELS, VALUE_LAYER_CNT)
        self.value = nn.Sequential(
                nn.Conv2d(CONV_CHANNELS, 1, kernel_size=1),
                nn.AvgPool2d(SHAPE),
                nn.Tanh()
                )

    def conv_layers(self, channels_in, channels_out, cnt):
            layers = []
            for _ in range(cnt):
                layers.append(nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False))
                channels_in = channels_out
                layers.append(nn.BatchNorm2d(channels_in))
                layers.append(nn.ReLU())

            return nn.Sequential(*layers)

    def forward(self, boards):
        # computing forward pass
        front  = self.front(boards)
        policy = self.policy(self.policy_conv(front))
        policy = F.log_softmax(policy.view(policy.shape[0],-1), dim=1)
        value  = self.value(self.value_conv(front))

        return value, policy
