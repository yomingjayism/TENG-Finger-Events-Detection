'''
Created on Fri Dec 13 10:51:37 2024

@author: Kendrick
'''

import os, sys
sys.path.insert(0, os.getcwd())
import glob
import numpy as np
from utils.io_utils import load_labeled_data, write_pickle
from config import TENGDataConfig

data_config = TENGDataConfig(window_size=100)

def slice_dataset(src:str, trg:str):
    filenames = glob.glob(os.path.join(src, "**"))
    target_dir = trg
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    for fpath in filenames:
        print(fpath)
        data = load_labeled_data(fpath)
        raw_signal = data["raw_signal"]
        normalized_signal = data["normalized_signal"]
        labels = data["labels"]
        
        signal_length, num_fingers = raw_signal.shape
        
        file_index = 0
        for i_finger in range(num_fingers):
            ptr = 0
            while ptr < (signal_length-1):
                data_pairs = {}
                end_idx = ptr + data_config.window_size
                
                raw_signal_seg = raw_signal[ptr:end_idx, i_finger].reshape(-1, 1)
                normalized_signal_seg = normalized_signal[ptr:end_idx, i_finger].reshape(-1, 1)
                label_seg = labels[ptr:end_idx, i_finger].reshape(-1, 1)
                if np.all(label_seg == 0):
                    ptr += data_config.stride
                    continue
                
                file_index += 1
                data_pairs["raw_signal"] = raw_signal_seg
                data_pairs["normalized_signal"] = normalized_signal_seg
                data_pairs["labels"] = label_seg
                
                save_path = os.path.join(f"{target_dir}", f"{file_index:05}.p")
                write_pickle(save_path, data_pairs)
                ptr += data_config.stride

        
if __name__ == "__main__":
    slice_dataset("labeled_data/train", "independent_dataset/train")
    slice_dataset("labeled_data/valid", "independent_dataset/valid")