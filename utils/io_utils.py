'''
Created on Thu Dec 12 09:54:32 2024

@author: Kendrick
'''

import numpy as np
import pandas as pd
import pickle

def load_raw_data(path, col_names=None):
    signal = pd.read_excel(path)
    if col_names is None:
        return signal
    data = []
    for c_name in col_names:
        data.append(np.array(signal[c_name]))
    
    data = np.array(data)
    data[np.isnan(data)] = 0 # nan handling
    
    return data.T

def load_labeled_data(path:str, num_fingers:int=3) -> dict:
    df = pd.read_csv(path)
    
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
    
    data = {"raw_signal": None, "normalized_signal": None, "labels": None}
    keys = ["raw_signal", "normalized_signal", "labels"]
    signals = []
    for c_name in dim_names:
        signals.append(np.array(df[c_name])[:, np.newaxis])
    
    signals = np.concatenate(signals, axis=1)
    for i, key in enumerate(keys):
        data[key] = signals[:, i * num_fingers:(i+1) * num_fingers]
        
    return data

def read_pickle(path:str):
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data

def write_pickle(path:str, data) -> None:
    with open(path, "wb") as fp:
        pickle.dump(data, fp)
    return