'''
Created on Thu Dec 12 10:05:59 2024

@author: Kendrick
'''
import numpy as np
from scipy.signal import find_peaks

def _find_pos_neg_seg(sign):
    segs = []
    seg = np.arange(len(sign))
    diff = np.abs(sign[1:] - sign[:-1])
    diff_indice = np.where(diff!=0)[0] + 1
    if len(diff_indice) == 0:
        segs.append(seg)
        return segs
    else:
        
        for idx in range(len(diff_indice)):
            if idx == 0:
                segs.append(seg[:diff_indice[idx]])
            else:
                segs.append(seg[diff_indice[idx-1] : diff_indice[idx]])
        return segs
    
def _find_threshold(pos_peak_values, neg_peak_values):
    pos_max_val = np.max(pos_peak_values)
    pos_min_val = np.min(pos_peak_values)
    neg_max_val = np.max(neg_peak_values)
    neg_min_val = np.min(neg_peak_values)
    
    pos_thres = np.mean([pos_max_val, pos_min_val])
    neg_thres = np.mean([neg_max_val, neg_min_val])
    return pos_thres, neg_thres

def labeling(signal:np.ndarray, start_thres:float=0.1):
    n_channel = signal.shape[-1]
    max_val, min_val = np.max(signal, axis=0), np.min(signal, axis=0)
    
    # normalize to -1 ~ 1
    signal = (signal - min_val.reshape(1, -1)) / (max_val - min_val).reshape(1, -1)
    signal = (signal - 0.5)  * 2
    peaks = np.zeros((signal.shape[0], n_channel))
    labels = np.zeros((signal.shape[0], n_channel))
    
    # Remove tiny signal
    pos_signal = signal.copy()
    neg_signal = signal.copy()
    pos_signal[signal < start_thres] = 0
    neg_signal[-signal < start_thres] = 0
    signal = pos_signal + neg_signal
    
    for i_ch in range(n_channel):
        pos_peaks, _ = find_peaks(signal[:, i_ch], height=start_thres, distance=10, prominence=0.2, width=5)
        neg_peaks, _ = find_peaks(-signal[:, i_ch], height=start_thres, distance=10, prominence=0.2, width=5)
        peaks[pos_peaks, i_ch] = 1
        peaks[neg_peaks, i_ch] = -1
    
    for i_ch in range(n_channel):
        peak_indice = np.where(peaks[:, i_ch]!=0)[0]
        outliers_indice = []
        outliers_sign = []
        last_peak = 0
        for i, peak_index in enumerate(peak_indice):
            if last_peak == 0:
                if peaks[peak_index, i_ch] != 1:
                    outliers_indice.append(peak_index)
                    outliers_sign.append(peaks[peak_index, i_ch])
                else:
                    last_peak = peaks[peak_index, i_ch]
            
            elif last_peak == 1:
                if peaks[peak_index, i_ch] != -1:
                    outliers_indice.append(peak_indice[i-1])
                    outliers_sign.append(peaks[peak_indice[i-1], i_ch])
                    outliers_indice.append(peak_index)
                    outliers_sign.append(peaks[peak_index, i_ch])
                else:
                    last_peak = peaks[peak_index, i_ch]
                    
            elif last_peak == -1:
                if peaks[peak_index, i_ch] != 1:
                    outliers_indice.append(peak_indice[i-1])
                    outliers_sign.append(peaks[peak_indice[i-1], i_ch])
                    outliers_indice.append(peak_index)
                    outliers_sign.append(peaks[peak_index, i_ch])
                else:
                    last_peak = peaks[peak_index, i_ch]
                    
        outliers_indice = np.array(outliers_indice)
        outliers_sign = np.array(outliers_sign)
        segs = _find_pos_neg_seg(outliers_sign)
        for seg in segs:
            if len(seg) == 0:
                continue
            outliers_indice = np.delete(outliers_indice, np.argmax(signal[outliers_indice[seg], i_ch])) # max response
            
        if len(outliers_indice) != 0:
            peaks[outliers_indice, i_ch] = 0
            
        pos_peaks = np.where(peaks[:, i_ch]>0)[0]
        neg_peaks = np.where(peaks[:, i_ch]<0)[0]
        pos_peaks_value = signal[pos_peaks, i_ch]
        neg_peaks_value = signal[neg_peaks, i_ch]
        pos_thres, neg_thres = _find_threshold(pos_peaks_value, neg_peaks_value)
        
        for peak in pos_peaks:
            labels[peak, i_ch] = 1 if signal[peak, i_ch] >= pos_thres else 2
        for peak in neg_peaks:
            labels[peak, i_ch] = 3 if signal[peak, i_ch] < neg_thres else 4
        
        pos_thres = ((pos_thres / 2 + 0.5) * (max_val - min_val)[i_ch]) + min_val[i_ch]
        neg_thres = ((neg_thres / 2 + 0.5) * (max_val - min_val)[i_ch]) + min_val[i_ch]
    return signal, peaks, labels