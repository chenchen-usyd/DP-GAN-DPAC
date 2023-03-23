import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['mlp']

class MLP(nn.Module):
    def __init__(self, input_size=32, nc=3, num_classes=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(np.prod((nc, input_size, input_size)), 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def mlp(**kwargs):
    return MLP(**kwargs)
