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
signal_no = 1
ecg_signal = df.iloc[:,signal_no][0:500]
print(ecg_signal)
plt.plot(ecg_signal)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Read the DataFrame
df = pd.read_csv('ECG_signals_col.csv')

# Number of signals you want to plot
num_signals = 5

# Select the signals you want to plot
signals = df.iloc[:, :num_signals]

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=num_signals, figsize=(15, 5))

# Plot each signal
for i in range(num_signals):
    axes[i].plot(signals.iloc[:, i][0:500])
    axes[i].set_title(f'Signal {i}')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()

