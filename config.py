'''
Created on Thu Dec 12 13:06:46 2024

@author: Kendrick
'''

from dataclasses import dataclass

@dataclass
class TENGDataConfig():
    window_size:int=100
    stride:int=10
    num_fingers:int=3
    gaussian_bins:int=19
    
@dataclass
class GestureWaveModelConfig():
    in_channels:int
    dims:list
    out_channels:int
    model_name:str=""
    
inference_config = TENGDataConfig(window_size=100, gaussian_bins=19)
