import torch.nn as nn
import torch.nn.functional as F

__all__ = ['cnn']

class CNN(nn.Module):
    def __init__(self, input_size=32, nc=3, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(nc, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout2d(0.5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out)        
        out = self.relu(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def cnn(**kwargs):
    return CNN(**kwargs)
