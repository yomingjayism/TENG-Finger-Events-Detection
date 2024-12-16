'''
Created on Fri Dec 13 09:37:35 2024

@author: Kendrick
'''
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from torch.utils.data import DataLoader
from config import TENGDataConfig, GestureWaveModelConfig
from utils.data import TENGestureDataset
from models.model import GestureWaveConv, GestureWaveHybrid
import matplotlib.pyplot as plt

data_config = TENGDataConfig(
    window_size=100,
    stride=10,
    num_fingers=3,
    gaussian_bins=19)

model_name = "gesturewave_conv_100_independent_1"
model_config = torch.load(model_name+".pt")["config"]

model = GestureWaveConv(config=model_config)
model.load_state_dict(torch.load("checkpoints/" + model_config.model_name+".pt")["checkpoints"])
model.eval()

validset = TENGestureDataset(
    "independent_dataset/valid",
    data_config
)

validloader = DataLoader(validset, 1, shuffle=True)
for i, data in enumerate(validloader):
    signal = data[0].float().permute(0, 2, 1)
    label = data[1].view(1, 100, -1).float()
    with torch.no_grad():
        pred = model(signal)["pred"].permute(0, 2, 1)
    
    signal = signal[0].permute(1, 0).numpy()
    pred = pred.numpy()[0, :, :4]
    label = label.numpy()[0, :, :4]
    
    fig = plt.figure(figsize=(18, 5), dpi=125)
    ax1 = plt.subplot2grid((5,2), (0, 0))
    ax2 = plt.subplot2grid((5,2), (1, 0))
    ax3 = plt.subplot2grid((5,2), (2, 0))
    ax4 = plt.subplot2grid((5,2), (3, 0))
    ax5 = plt.subplot2grid((5,2), (4, 0))
    ax6 = plt.subplot2grid((5,2), (0, 1))
    ax7 = plt.subplot2grid((5,2), (1, 1))
    ax8 = plt.subplot2grid((5,2), (2, 1))
    ax9 = plt.subplot2grid((5,2), (3, 1))
    ax10 = plt.subplot2grid((5,2), (4, 1))
    
    ax1.set_title(f"Signal")
    ax1.set_ylim(-1, 1)
    ax1.plot(signal[:, 0])
    
    ax2.set_title(f"Label 1")
    ax2.set_ylim(0, 1)
    ax2.plot(label[:, 0])
    
    ax3.set_title(f"Label 2")
    ax3.set_ylim(0, 1)
    ax3.plot(label[:, 1])
    
    ax4.set_title(f"Label 3")
    ax4.set_ylim(0, 1)
    ax4.plot(label[:, 2])
    
    ax5.set_title(f"Label 4")
    ax5.set_ylim(0, 1)
    ax5.plot(label[:, 3])
    
    ax6.set_title(f"Signal")
    ax6.set_ylim(-1, 1)
    ax6.plot(signal[:, 0])
    
    ax7.set_title(f"Predict 1")
    ax7.set_ylim(0, 1)
    ax7.plot(pred[:, 0])
    
    ax8.set_title(f"Predict 2")
    ax8.set_ylim(0, 1)
    ax8.plot(pred[:, 1])
    
    ax9.set_title(f"Predict 3")
    ax9.set_ylim(0, 1)
    ax9.plot(pred[:, 2])
    
    ax10.set_title(f"Predict 4")
    ax10.set_ylim(0, 1)
    ax10.plot(pred[:, 3])
    
    plt.tight_layout()
    plt.show()
    
    