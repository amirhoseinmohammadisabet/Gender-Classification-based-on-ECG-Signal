import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# Define the denoising functions
def remove_baseline_wander(ecg_signal, fs=1000):
    b, a = signal.butter(4, 1 / (fs / 2), btype='high')
    return signal.filtfilt(b, a, ecg_signal)

def remove_powerline_interference(ecg_signal, fs=1000):
    f0 = 50  # or 60 Hz
    Q = 30.0  # Quality factor
    w0 = f0 / (fs / 2)  # Normalized frequency
    b, a = signal.iirnotch(w0, Q)
    return signal.filtfilt(b, a, ecg_signal)

def remove_high_frequency_noise(ecg_signal, fs=1000):
    b, a = signal.butter(4, 100 / (fs / 2), btype='low')
    return signal.filtfilt(b, a, ecg_signal)

# Load your ECG signal from the CSV file
# Assuming the signal is in the first column of the CSV file
df = pd.read_csv('ECG_signals_col.csv')

ecg_signal = df.iloc[:,1][0:500]
print(ecg_signal)
plt.plot(ecg_signal)
plt.show()