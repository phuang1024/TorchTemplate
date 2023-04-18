import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from constants import *


class MyDataset(Dataset):
    def __init__(self, directory):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
