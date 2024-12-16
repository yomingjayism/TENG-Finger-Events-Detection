'''
Created on Thu Dec 12 15:34:39 2024

@author: Kendrick
'''
import os, sys
sys.path.insert(0, os.getcwd())

import torch
import torch.optim as optim
from models.model import GestureWaveConv
from utils.data import TENGestureDataset
from utils.trainer import Trainer
from config import TENGDataConfig, GestureWaveModelConfig

data_config = TENGDataConfig(
    window_size=100,
    stride=10,
    num_fingers=3,
    gaussian_bins=19)

model_config = GestureWaveModelConfig(
    model_name="gesturewave_conv_100_independent_1_bill",
    in_channels=1,
    dims=[16, 32, 64],
    out_channels=4,
)

trainset = TENGestureDataset(
    "independent_dataset/train",
    data_config
)

validset = TENGestureDataset(
    "independent_dataset/valid",
    data_config
)

model = GestureWaveConv(config=model_config)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cpu")
trainer = Trainer(
    batch_size=32,
    num_epoch=40,
    trainset=trainset,
    validset=validset,
    optimizer=optimizer,
    model=model,
    device=device
)

trainer.train()