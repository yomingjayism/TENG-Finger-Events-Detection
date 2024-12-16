'''
Created on Fri Dec 13 16:19:54 2024

@author: Kendrick
'''
import torch
import os, sys
sys.path.insert(0, os.getcwd())
from config import TENGDataConfig
from utils.io_utils import load_labeled_data
from models.model import GestureWaveConv
from models.detector import EventDetector

model_name = "gesturewave_conv_100_independent_1_bill"
model_path = os.path.join("checkpoints", model_name)
model_config = torch.load(model_path+".pt")["config"]
model = GestureWaveConv(config=model_config)
model.load_state_dict(torch.load(model_path+".pt")["checkpoints"])
model.eval()

detector = EventDetector(
    model=model,
    window_size=100,
    num_fingers=3,
    event_num=4
)

data_config = TENGDataConfig(
    window_size=100,
    stride=10,
    num_fingers=3,
    gaussian_bins=19)

data = load_labeled_data(
    "labeled_data/valid/Bill2_6.csv"
)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    
    signals = data["raw_signal"].T[:, np.newaxis, :]
    labels = data["labels"]
    ptr = 0
    signal_len = signals.shape[-1]
    global_predictions = np.zeros((data_config.num_fingers, signal_len))
    
    while ptr < signal_len:
        try:
            start_time = time.time()
            
            end_idx = ptr + data_config.window_size
            
            signal_data = signals[:, :, ptr:end_idx] # (N * 1 * l)
            events = detector.detect(signal_data)
            events_indice = np.where(events != 0)
            time_index = events_indice[-1]
            if len(time_index) >0:
                max_step = max(time_index)
                ptr += max_step + data_config.stride
            else:
                ptr += data_config.stride
            
            global_predictions[events_indice[0], ptr+events_indice[1]] = events[events_indice]
            print(ptr, "/", signal_len)
            
        except Exception as e:
            print(e)
            ptr = signal_len
            pass
    
    fig = plt.figure(figsize=(18, 5), dpi=125)
    ax1 = plt.subplot2grid((4,2), (0, 0))
    ax2 = plt.subplot2grid((4,2), (1, 0))
    ax3 = plt.subplot2grid((4,2), (2, 0))
    ax4 = plt.subplot2grid((4,2), (3, 0))
    ax5 = plt.subplot2grid((4,2), (0, 1))
    ax6 = plt.subplot2grid((4,2), (1, 1))
    ax7 = plt.subplot2grid((4,2), (2, 1))
    ax8 = plt.subplot2grid((4,2), (3, 1))
    
    
    ax1.set_title(f"Signal")
    ax1.set_ylim(-1, 1)
    ax1.plot(data["raw_signal"])
    
    ax2.set_title(f"Label 1")
    ax2.set_ylim(0, 4)
    ax2.plot(labels[:, 0])
    
    ax3.set_title(f"Label 2")
    ax3.set_ylim(0, 4)
    ax3.plot(labels[:, 1])
    
    ax4.set_title(f"Label 3")
    ax4.set_ylim(0, 4)
    ax4.plot(labels[:, 2])
    
    ax5.set_title(f"Signal")
    ax5.set_ylim(-1, 1)
    ax5.plot(data["raw_signal"])
    
    ax6.set_title(f"Prediction")
    ax6.set_ylim(0, 4)
    ax6.plot(global_predictions[0, :])
    
    ax7.set_title(f"Prediction")
    ax7.set_ylim(0, 4)
    ax7.plot(global_predictions[1, :])
    
    ax8.set_title(f"Prediction")
    ax8.set_ylim(0, 4)
    ax8.plot(global_predictions[2, :])
    
    plt.tight_layout()
    plt.show()