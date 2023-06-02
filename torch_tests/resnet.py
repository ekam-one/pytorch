import torch
import torch.nn as nn
import torch.nn.functional as F


import torchvision

class ResNetKernel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + x

model = torchvision.models.resnet18()

# Compile the model
model = torch.compile(model, backend="torchmhlo")

# Make a prediction
input_data = torch.randn(1, 3, 224, 224)
prediction = model(input_data)