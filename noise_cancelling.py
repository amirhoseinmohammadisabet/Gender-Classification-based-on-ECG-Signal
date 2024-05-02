import numpy as np
import scipy.io as sio
import pywt
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pandas as pd




def ecgfilter(signal):
    # Butterworth filter parameters
    cutoff_freq = 0.5  # Adjust cutoff frequency as needed
    order = 4  # Adjust filter order as needed

    # Design a low-pass Butterworth filter
    b, a = butter(order, cutoff_freq, btype='low')

    # Apply the filter to the signal using filtfilt for zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def denoisecg(signal, threshold):
    wavelet = 'db4'
    coeffs = pywt.wavedec(signal, wavelet)
    coeffs = [pywt.threshold(detail, threshold) if i > 0 else detail for i, detail in enumerate(coeffs)]
    denoised_signal = pywt.waverec(coeffs, wavelet)

    return denoised_signal


def plot_ecg(ecgsig, filtered_ecg, sample_rate=1):
    num_ecgs, num_samples = ecgsig.shape
    time = np.arange(0, num_samples / sample_rate, 1 / sample_rate)

    plt.figure(figsize=(10, 6))
    for i in range(num_ecgs):
        plt.plot(time, ecgsig[i], label=f'ECG {i + 1} (Original)', alpha=0.5)
        plt.plot(time, filtered_ecg[i], label=f'ECG {i + 1} (Filtered)')

    plt.title('ECG Signals Before and After Filtering')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


fs = 1000
ecg_signals = pd.read_csv('ECG_signals_col.csv')
ecg_signal = ecg_signals.iloc[:,5][0:500]
time = np.arange(len(ecg_signal))/fs


ecg_filtered_1 = ecgfilter(ecg_signal)
ecg_filtered_2 = denoisecg(ecg_signal,0.1)

import matplotlib.pyplot as plt

# Define the time array
time = np.arange(len(ecg_signal)) / fs

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time, ecg_signal, label='Original ECG Signal')
plt.plot(time, ecg_filtered_1, label='Filtered ECG Signal')
plt.plot(time, ecg_filtered_2, label='Denoised ECG Signal')
plt.title('ECG Signals Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()