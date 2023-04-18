import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters

# Training parameters
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
LR_DECAY_FAC = 0.9
LR_DECAY_STEPS = 5000
