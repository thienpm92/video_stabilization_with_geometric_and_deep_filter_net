import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def conv_layer(in_ch, out_ch, window=150, kernel=3, stride=1, padding=1,pooling=True):
    if pooling:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, bias=True, padding=padding),
            nn.AdaptiveAvgPool1d(window),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
    else:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, bias=True, padding=padding),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

class FilterNet(nn.Module):
    def __init__(self):
        super(FilterNet,self).__init__()
        self.layer1 = conv_layer(in_ch=1, out_ch=16, window=150, kernel=50, stride=1, padding=1,pooling=True)
        self.layer2 = conv_layer(in_ch=16, out_ch=32, window=150, kernel=30, stride=1, padding=1,pooling=True)
        self.layer3 = conv_layer(in_ch=32, out_ch=64, window=150, kernel=15, stride=1, padding=1,pooling=True)
        self.layer4 = conv_layer(in_ch=64, out_ch=128, window=150, kernel=10, stride=1, padding=1,pooling=True)
        self.layer5 = conv_layer(in_ch=128, out_ch=256, window=150, kernel=5, stride=1, padding=1,pooling=True)
        self.layer6 = conv_layer(in_ch=256, out_ch=128, window=150, kernel=10, stride=1, padding=1,pooling=True)
        self.layer7 = conv_layer(in_ch=128, out_ch=64, window=150, kernel=15, stride=1, padding=1,pooling=True)
        self.layer8 = conv_layer(in_ch=64, out_ch=32, window=150, kernel=15, stride=1, padding=1,pooling=True)
        self.layer9 = conv_layer(in_ch=32, out_ch=16, window=150, kernel=30, stride=1, padding=1,pooling=True)
        self.layer10 = conv_layer(in_ch=16, out_ch=1, window=150, kernel=30, stride=1, padding=1,pooling=True)

    def forward(self,inputs):
        x1 = self.layer1(inputs)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)
        outputs = self.layer10(x9)
        return outputs


