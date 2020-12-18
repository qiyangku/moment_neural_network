# -*- coding: utf-8 -*-
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from Mnn_Core.mnn_pytorch import *
from torch.utils.tensorboard import SummaryWriter


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Mnn_Alex_Net(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(Mnn_Alex_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = Mnn_Linear_without_Corr(256 * 6 * 6, 2096, bias=True)
        self.fc2 = Mnn_Linear_without_Corr(4096, 4096, bias=True)
        self.fc3 = Mnn_Linear_without_Corr(4096, num_classes, bias=True)
        self.a1 = Mnn_Activate_Mean.apply
        self.a2 = Mnn_Activate_Std.apply
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        y = torch.sqrt(torch.abs(x))

        x = self.dropout(x)
        temp = torch.zeros_like(y)
        y = torch.where(x == 0, temp, y)
        x, y = self.fc1(x, y)
        x_a = self.a1(x, y)
        y_a = self.a2(x, y, x_a)

        x = self.dropout(x_a)
        y = torch.where(x == 0, temp, y_a)
        x, y = self.fc2(x, y)
        x_a = self.a1(x, y)
        y_a = self.a2(x, y, x_a)
        x, y = self.fc3(x_a, y_a)
        return x


model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
model.eval()

for name in model.state_dict():
    print(model.state_dict()[name].shape)
