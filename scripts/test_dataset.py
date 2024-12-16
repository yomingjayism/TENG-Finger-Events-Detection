'''
Created on Thu Dec 12 14:25:56 2024

@author: Kendrick
'''
import os, sys
sys.path.insert(0, os.getcwd())
from utils.data import TENGestureDataset
from config import TENGDataConfig
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_config = TENGDataConfig()
    trainset = TENGestureDataset("./independent_dataset/train", data_config)
    trainloader = DataLoader(trainset, 1, shuffle=True)
    
    for i, data in enumerate(trainloader):
        signals = data[0]
        labels = data[1]
        pad_mask = data[2]
        
        fig = plt.figure(figsize=(18, 5), dpi=125)
        
        plt.subplot(511)
        plt.title(f"Thumb Signal")
        plt.ylim(-1, 1)
        plt.plot(signals.squeeze(0)[:, 0])
        
        plt.subplot(512)
        plt.title(f"event1 label")
        plt.ylim(0, 1)
        plt.plot(labels.squeeze(0)[:, 0, 0])
        
        plt.subplot(513)
        plt.title(f"event2 label")
        plt.ylim(0, 1)
        plt.plot(labels.squeeze(0)[:, 0, 1])
        
        plt.subplot(514)
        plt.title(f"event3 label")
        plt.ylim(0, 1)
        plt.plot(labels.squeeze(0)[:, 0, 2])
        
        plt.subplot(515)
        plt.title(f"event4 label")
        plt.ylim(0, 1)
        plt.plot(labels.squeeze(0)[:, 0, 3])
        plt.show()
        