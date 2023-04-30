import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
# TODO

# Model hyperparameters
# TODO

# Training parameters
BATCH_SIZE = 64
BATCH_PER_STEP = 1
EPOCHS = 5
SAVE_INTERVAL = 5
LR_START = 1e-3
LR_END = 1e-5
