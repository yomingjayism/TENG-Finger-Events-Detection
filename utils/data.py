'''
Created on Thu Dec 12 13:56:00 2024

@author: Kendrick
'''

import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import glob
from torch.utils.data import Dataset
from scipy.signal import windows
from config import TENGDataConfig
from utils.io_utils import read_pickle

def distribute(event_index:np.ndarray, window_size:int, bins:int) -> np.ndarray:
    """ Give onehot events segment and transform the event to distribution.
    
    Args:
        event_index: N*1 array, the event index
        bins: This denotes the width of guassian window.

    Returns:
        dist: N*1 array, the distribution of the event segment.

    """
    dist = np.zeros((window_size) + bins)
    
    sigma = (bins + 1)//2
    hist = windows.gaussian(bins, std=5)
    event_index = event_index + sigma
    for mean in event_index:
        dist[mean-sigma+1:mean+sigma] = hist
    
    dist = dist[sigma:-sigma+1]

    return dist

def transform_distribute(labels:np.ndarray, window_size:int, bins:int) -> np.ndarray:
    """ Transform the multi-channels labels to multi-channels guassian confidence score.
    
    Args:
        labels: The original labels with shape (sequence length, num_fingers).
        bins: The width of the guassian window.
    Returns:
        score_labels: The transformed labels with shape (num_channel, sequence length, 4).
    """
    num_fingers = labels.shape[1]
    score_labels = np.zeros((window_size, num_fingers, 4))
    for i_channel in range(num_fingers):
        label = labels[:, i_channel]
        label1_index = np.where(label==1)[0]
        label2_index = np.where(label==2)[0]
        label3_index = np.where(label==3)[0]
        label4_index = np.where(label==4)[0]
        
        label1_score = distribute(label1_index, window_size, bins).reshape(window_size, 1)
        label2_score = distribute(label2_index, window_size, bins).reshape(window_size, 1)
        label3_score = distribute(label3_index, window_size, bins).reshape(window_size, 1)
        label4_score = distribute(label4_index, window_size, bins).reshape(window_size, 1)
        label = np.concatenate([label1_score, label2_score, label3_score, label4_score], axis=1)
        score_labels[:, i_channel] = label
        
    return score_labels

class TENGestureDataset(Dataset):
    def __init__(self, src_path:str, config:TENGDataConfig) -> None:
        self.src_path = src_path
        self.filenames = glob.glob(os.path.join(src_path, "**"))
        self.config = config
        self.window_size = config.window_size
        
    def __len__(self) -> int:
        return len(self.filenames)
    
    def pad_seq(self, signal_seg:np.ndarray, label_seg:np.ndarray) -> tuple:
        # padding the sequence with zero padding if the length is NOT satisfied.
        if signal_seg.shape[0] < self.config.window_size:
            pad_len = self.config.window_size - signal_seg.shape[0]
            pad_seq = np.zeros((pad_len, signal_seg.shape[-1]))
            pad_mask = np.concatenate((np.ones((signal_seg.shape[0], 1)), 
                                      np.zeros((pad_seq.shape[0], 1))), axis=0)
            
            signal_seg = np.concatenate((signal_seg, pad_seq), axis=0)    
            label_seg = np.concatenate((label_seg, pad_seq), axis=0)
        else:
            pad_mask = np.ones((signal_seg.shape[0], 1))
            
        return signal_seg, label_seg, pad_mask
            
    def __getitem__(self, index) -> tuple:
        fpath = self.filenames[index]
        data = read_pickle(fpath)
        
        signal = data["normalized_signal"]
        labels = data["labels"]
        signal, labels, pad_mask = self.pad_seq(signal, labels)
        
        # transform to guassian
        scores = transform_distribute(labels, window_size=self.config.window_size, bins=self.config.gaussian_bins)
        return signal, scores, pad_mask