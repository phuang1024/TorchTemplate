import torch
import torch.nn as nn

from constants import *


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
