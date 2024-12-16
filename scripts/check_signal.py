'''
Created on Fri Dec 13 10:12:54 2024

@author: Kendrick
'''
import os, sys
sys.path.insert(0, os.getcwd())

from utils.io_utils import load_labeled_data
import matplotlib.pyplot as plt
import glob
import numpy as np

filenames = glob.glob("labeled_data/**")
check_size = 150
check_window = np.zeros((check_size,))

for fpath in filenames:
    print(fpath)
    fig = plt.figure(figsize=(18, 5), dpi=125)
    data = load_labeled_data(fpath)
    
    labels = data["labels"]
    seq_len = len(labels)
                
    plt.subplot(321)
    plt.title(f"raw signal 1")
    plt.plot(data["raw_signal"][:, 0])
    
    plt.subplot(323)
    plt.title(f"raw signal 2")
    plt.plot(data["raw_signal"][:, 1])
    
    plt.subplot(325)
    plt.title(f"raw signal 3")
    plt.plot(data["raw_signal"][:, 2])
    
    plt.subplot(322)
    plt.title(f"norm signal 1")
    plt.plot(data["normalized_signal"][:, 0])
    
    plt.subplot(324)
    plt.title(f"norm signal 2")
    plt.plot(data["normalized_signal"][:, 1])
    
    plt.subplot(326)
    plt.title(f"norm signal 3")
    plt.plot(data["normalized_signal"][:, 2])
    
    plt.tight_layout()
    plt.show()
