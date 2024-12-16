'''
Created on Thu Dec 12 10:09:14 2024

@author: Kendrick
'''
import os, sys
sys.path.insert(0, os.getcwd())
import glob
import numpy as np
import pandas as pd
from utils.io_utils import load_raw_data
from utils.labeling import labeling

save_dir = "./labeled_data"
col_names = [f"Voltage (V) - Plot {i}" for i in range(3)] # three fingers
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
if __name__ == "__main__":
    f_filenames = sorted(glob.glob("raw_data/RingSensorData_1120/**/f/**"))
    h_filenames = sorted(glob.glob("raw_data/RingSensorData_1120/**/h/**"))
    
    subject = None
    subject_count = 0
    dim_names = [
        "thumb_raw_signal", 
        "index_raw_signal", 
        "middle_raw_signal",
        "thumb_normalized_signal", 
        "index_normalized_signal", 
        "middle_normalized_signal",
        "thumb_label", 
        "index_label", 
        "middle_label",
        ]
    
    for i, (f_path, h_path) in enumerate(zip(f_filenames, h_filenames)):
        print(f"{f_path} || {h_path}")
        data1 = load_raw_data(h_path, col_names)
        data2 = load_raw_data(f_path, col_names)
        
        if subject is None:
            subject = f_path.split("/")[2]
            subject_count = 1
        else:
            temp_subject = f_path.split("/")[2]
            if subject == temp_subject:
                subject_count += 1
            else:
                subject = temp_subject
                subject_count =1
            
        data = np.vstack((data1, data2))
        _, peaks, label = labeling(data, 0.05)
        
        
        max_val, min_val = np.max(data, axis=0), np.min(data, axis=0)
        
        # normalize to -1 ~ 1 wit min-max normalization
        signal = (data - min_val.reshape(1, -1)) / (max_val - min_val).reshape(1, -1)
        signal = (signal - 0.5)  * 2
        
        save_path = os.path.join(save_dir, f"{subject}_{subject_count}.csv")
        
        all_data = np.hstack((data, signal, label))
        dataframe = {}
        for i_dim, dim in enumerate(dim_names):
            dataframe[dim] = all_data[:, i_dim].tolist()
        
        df = pd.DataFrame(dataframe)
        df.to_csv(save_path)
        