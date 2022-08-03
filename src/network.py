import json

import torch
import torch.nn as nn


with open("settings.json", "r") as f:
    settings = json.load(f)


class Network(nn.Module):
    """
    Example MNIST digit network.
    """

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.main(x)
