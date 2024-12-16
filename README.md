# TENG-Finger-Event-Detection

This a repository for the TENG-Ring-Sensor Event Detection based on machine learning model. With this event detector, we can achieve both realtime and offline detecting technically.


## Requirements
matplotlib==3.5.1\
numpy==1.26.4\
pandas==2.2.2\
scipy==1.14.1\
torch==2.0.1+cpu\
tqdm==4.66.4

Run `pip install -r requirements.txt` or `pip3 install -r requirements.txt` to install all dependencies.

## How to use it?

In `run_detector.py` we load a record(*.csv) and simulate the situation when data are streaming in real-time, we use a sliding window to contingously feed the data into the model and we collect the inference result at every moment then visualize the final result for this record.

We firstly load the model and prepare the data. Here you can see we set some configurations.
```python

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

```

And we feed the data into the model with a sliding window to simulate the real-time inference.
```python

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

```

Use `python run_detector.py` to see detailed result.

### Inputs & Outputs
The detector inputs should be the signal with the shape `(N * 1 * l)`. Where N denotes the number of the finger at the inference stage. l denotes the length of the signal.\
The detector outputs would be the prediction with the shape `(N * L)`. Where N denotes the number of the finger at the inference stage. L denotes the window size.

## Model Training

We use a tiny model which makes training on cpu machine possible, and for better model performance, we recommended training individual models for each subject. 

To train the model from scratch, you will need to split the data(*.csv) in `./labeled_data` into training and valdating sets. For examples, if you'd like to train the model for Bill, you can copy the files: Bill2_1.csv, Bill2_2.csv, Bill2_3.csv, Bill2_4.csv, Bill2_5.csv to `./labeld_data/train` for training data, and then, copy Bill2_6.csv to `./labeled_data/valid` as the validating data.

Run `python ./scripts/slice_data_independent.py` to slice the data into training pairs.

And run `python ./scripts/train_gesturewave_conv.py` to train the model.


## Configuration

Note that there might be some difference between the config for training and the config for the detector.

For training config, cause all the training data we have are from three fingers, we always have num_fingers=3. However, we will have num_fingers=6 even more in real case when detecting the finer events. It is important to modify the configuration to fit either two situations.

We explain each configuration parameters as follows:

For ``TENGDataConfig``:\
`window_size`: determines the data length we feed into the model.\
`stride`: determines the step everytime the sliding window takes to move forward. Large step would speed up the inference but some events may not be detected.\
`num_fingers` determines the number of the fingers that used in current stage. for training, this can be set any number, but it should be considered in inference stage.
`gaussian_bins`: determines the width of the guassian template we used in training and post-processing. Wider bins means stronger uncertainty and label smoothing.


For ``GestureWaveModelConfig``:\
`in_channels` determines the channel of the signal. Here we set it 1, because we let the number_fingers as the "batch" instead of "channel".\
`dims` determines the dimensions of all hidden layers. It shoud be a list contains 3 integer elements.\
`out_channels` determins the channel of the model's output. We have four events, so it is set to 4.\
`model_name` is the name of the model, which would be taken by the trainer to save the checkpoints. Named the model whatever you want.

## Evaluation
Comming soon...

