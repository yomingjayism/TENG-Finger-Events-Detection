'''
Created on Fri Dec 13 15:44:17 2024

@author: Kendrick
'''

from scipy import signal as scisignal
from config import inference_config
import torch
import torch.nn as nn
import numpy as np

class EventDetector():
    def __init__(self, model:nn.Module, window_size:int=100, num_fingers:int=6, event_num:int=4):
        self.model = model
        self.window_size = window_size
        self.num_fingers = num_fingers
        self.event_num = event_num
        self.pattern = scisignal.windows.gaussian(
            inference_config.gaussian_bins, 
            std=5).astype("float32")
        
        self.pattern_width = inference_config.gaussian_bins // 2
        
    def detect(self, signals:np.ndarray, threshold:float=0.5) -> np.ndarray:
        """ Feed the TENG Ring Sensor Signal, then get the
            event index.
            
        Args:
            signals: The TENG signals. A (N * 1 * l) array;
                where N is number of fingers, l is the signal length.
                
            threshold: The threshold for IOU metrics selection. Defualt is 0.5.
            
        Returns:
            events: The results of the detected events index. Shape: (N * L).
        """
        with torch.no_grad():
            signals = torch.from_numpy(signals).float()
            signals = self.pad_(signals) # pad to the window size
            pred = self.model(signals)["pred"] #(N, 1, L)
            
        events = self.iou_match(pred, threshold).numpy()
        return events
            
    def iou_match(self, predict, threshold:float) -> tuple:
        """ Calculate the IOU(Intersection Over Union) score between 
            the prediction and gaussian template.
            
        Args:
            predict: The model prediction, confidence score. Shape: (N * 4 * L).
            threshold: The threshold for IOU metrics selection.
            
        Returns:
            events: The events data. Shape (N * L).
            iou_record: The iou score at every moment. Shape (N * L).
        """
        events = torch.zeros((self.num_fingers, self.window_size), dtype=torch.int32)
        iou_record = torch.zeros((self.num_fingers, self.window_size))
        
        pattern = torch.from_numpy(self.pattern).unsqueeze(0).unsqueeze(1).repeat(self.num_fingers, self.event_num, 1)
        
        for center_index in range(self.pattern_width, self.window_size-self.pattern_width):
            iou_data = predict[:, :, (center_index - self.pattern_width):(center_index + self.pattern_width + 1)]
            union = torch.maximum(iou_data, pattern)
            intrsct = torch.minimum(iou_data, pattern)
            iou_score = torch.mean(intrsct / (union + 1e-16), dim=-1) # (N, 4)
            max_iou_score = torch.max(iou_score, dim=1).values
            candidate_event_index = torch.argmax(iou_score, dim=-1)# (N,)
            for i_finger in range(self.num_fingers):
                events[i_finger, center_index] = candidate_event_index[i_finger] + 1 \
                    if iou_score[i_finger, candidate_event_index[i_finger]] >= threshold else 0
                
                iou_record[i_finger, center_index] = max_iou_score[i_finger]
                
        events = self._merge_events(events, iou_record)
        return events
    
    def _merge_events(self, events:torch.tensor, iou_record:torch.tensor) -> tuple:
        """ Merge the duplicate events.
        
        Args:
            events: The events data. Shape (N * L).
            iou_record: The iou score at every moment. Shape (N * L).
        """
        for i in range(self.num_fingers):
            last_event = -1
            last_index = 0
            last_iou_score = 0
            
            for t in range(self.window_size):
                events_t = events[i, t].clone()
                iou_record_t = iou_record[i, t].clone()
                if events_t == 0:
                    last_event = 0
                    last_index = t
                    last_iou_score = iou_record_t
                    continue
                
                elif events_t != last_event:
                    last_event = events_t
                    last_index = t
                    last_iou_score = iou_record_t
                    
                elif events_t == last_event:
                    if iou_record_t > last_iou_score:
                        events[i, last_index] = 0 # replace previous events with nothing happened.
                        last_index = t
                        last_iou_score = iou_record_t
                        
                    else:
                        events[i, t] = 0 # replace current events with nothing happened

        return events
    
    def pad_(self, signals:torch.tensor) -> torch.tensor:
        """ Pad zeros to the signal as the size as the target window size.
        
        Args:
            signal: The signal with shape (N * 1 * l). Where l denotes the singal size.
        Returns:
            padded_signal: The signal after padding to the window size length. With shape
                (N * 1 * L).
        
        """
        pad_len = self.window_size - signals.shape[-1]
        padded_seq = torch.zeros((self.num_fingers, 1, pad_len))
        padded_signal = torch.cat((signals, padded_seq), dim=-1)
        return padded_signal