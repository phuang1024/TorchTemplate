from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from constants import *


class MyDataset(Dataset):
    def __init__(self, directory: Path):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx):
        pass
