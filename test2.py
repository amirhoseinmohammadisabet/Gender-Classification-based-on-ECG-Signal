import numpy as np
import scipy.io as sio
import pywt
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


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


def ecgin(pernum, lengsig):
    ecgsig = np.zeros((pernum, lengsig))
    data_folder = "Data/MIT_BIH_NS"  # Specify the folder name
    for x in range(1, pernum + 1):
        filename = f"{x}.mat"
        filepath = os.path.join(data_folder, filename)
        data = sio.loadmat(filepath)
        ecgsig = data['val']
        # You can add filtering or denoising steps here if needed
    return ecgsig


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


# Assuming you have 3 ECG signals to process, each with a length of 10000 points
pernum = 3
lengsig = 10000

# Call ecgin function to process the ECG signals
ecg_signals = ecgin(pernum, lengsig)

# Apply filtering to the ECG signals
filtered_ecg_signals = np.array([ecgfilter(signal) for signal in ecg_signals])

# Plot original and filtered signals
plot_ecg(ecg_signals, filtered_ecg_signals, sample_rate=2)
