import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
import csv

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels

        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.bn2 = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = nn.ReLU()(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return nn.ReLU()(x + y)


class MY_NET(nn.Module):  # 从父类 nn.Module 继承
    def __init__(self):
        super(MY_NET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(64)
        self.res1 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(128)
        self.res2 = ResidualBlock(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(256)
        self.res3 = ResidualBlock(256)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dense = Sequential(
            nn.Linear(28 * 28 * 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 500)
        )
        torch.nn.init.xavier_normal_(self.dense[0].weight)
        torch.nn.init.xavier_normal_(self.dense[4].weight)


    def forward(self, x):  # 正向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.res1(x)
      
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.res2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.res3(x)

        x = x.view(-1, 28 * 28 * 256)
        x = self.dense(x)
        return x
