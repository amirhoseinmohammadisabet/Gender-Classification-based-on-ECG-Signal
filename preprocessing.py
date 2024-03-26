import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def ecgin(ax):
    # Load .mat ECG data
    mat_file_path = 'Data/Big/JS00001.mat'
    ecg_data = scipy.io.loadmat(mat_file_path)

    # Print the keys in the loaded data
    print("Keys in the loaded .mat file:", ecg_data.keys())

    # Automatically select the first variable as the ECG signal
    # You may need to adjust this if your .mat file has a different structure
    variable_names = list(ecg_data.keys())
    if variable_names:
        first_variable_name = variable_names[0]
        ecg_signal = ecg_data[first_variable_name]
        # print(ecg_signal[5][150:350])
        if ax == 1:
            # Plot the ECG signal
            plt.plot(ecg_signal[5])
            plt.title(f'ECG Signal ({first_variable_name})')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.show()
    else:
        print("No variables found in the loaded .mat file.")

    return ecg_signal


    
sample_ecg = ecgin(0)
ecg_signal = sample_ecg[5][150:5000] 



# Provided ECG signal


def remove_baseline_wander(ecg_signal, fs=1000):
    """
    Remove baseline wander from the ECG signal using a high-pass filter.

    Parameters:
    ecg_signal (numpy.ndarray): ECG signal.
    fs (int): Sampling frequency (Hz).

    Returns:
    numpy.ndarray: ECG signal with baseline wander removed.
    """
    # Design a 1 Hz high-pass Butterworth filter
    b, a = signal.butter(4, 1 / (fs / 2), btype='high')

    # Apply the filter to remove baseline wander
    filtered_signal = signal.filtfilt(b, a, ecg_signal)

    return filtered_signal


def remove_powerline_interference(ecg_signal, fs=1000):
    """
    Remove powerline interference (50 Hz or 60 Hz) from the ECG signal using a notch filter.

    Parameters:
    ecg_signal (numpy.ndarray): ECG signal.
    fs (int): Sampling frequency (Hz).

    Returns:
    numpy.ndarray: ECG signal with powerline interference removed.
    """
    # Design a notch filter to remove powerline interference
    f0 = 50  # or 60 Hz
    Q = 30.0  # Quality factor
    w0 = f0 / (fs / 2)  # Normalized frequency
    b, a = signal.iirnotch(w0, Q)

    # Apply the notch filter to remove powerline interference
    filtered_signal = signal.filtfilt(b, a, ecg_signal)

    return filtered_signal


def remove_high_frequency_noise(ecg_signal, fs=1000):
    """
    Remove high-frequency noise from the ECG signal using a low-pass filter.

    Parameters:
    ecg_signal (numpy.ndarray): ECG signal.
    fs (int): Sampling frequency (Hz).

    Returns:
    numpy.ndarray: ECG signal with high-frequency noise removed.
    """
    # Design a 100 Hz low-pass Butterworth filter
    b, a = signal.butter(3, 100 / (fs / 2), btype='low')

    # Apply the filter to remove high-frequency noise
    filtered_signal = signal.filtfilt(b, a, ecg_signal)

    return filtered_signal

def final_ecg_signal(ecg_signal):
    """
    Apply noise removal techniques sequentially to generate the final cleaned ECG signal.

    Parameters:
    ecg_signal (numpy.ndarray): Original ECG signal.

    Returns:
    numpy.ndarray: Cleaned ECG signal after applying noise removal techniques.
    """
    ecg_filtered_baseline = remove_baseline_wander(ecg_signal)
    ecg_filtered_powerline = remove_powerline_interference(ecg_filtered_baseline)
    ecg_filtered_noise = remove_high_frequency_noise(ecg_filtered_powerline)    
    return ecg_filtered_noise


def plot_noise_cancelling():
    # Remove baseline wander
    ecg_filtered_baseline = remove_baseline_wander(ecg_signal)

    # Remove powerline interference
    ecg_filtered_powerline = remove_powerline_interference(ecg_signal)

    # Remove high-frequency noise
    ecg_filtered_noise = remove_high_frequency_noise(ecg_signal)

    final_ecg_signals = final_ecg_signal(ecg_signal)
    # Plot the original and filtered signals
    plt.figure(figsize=(10, 6))
    plt.plot(ecg_signal, label='Original ECG')
    plt.plot(ecg_filtered_baseline, label='Baseline Removed', linestyle='--')
    plt.plot(ecg_filtered_powerline, label='Powerline Interference Removed', linestyle='-.')
    plt.plot(ecg_filtered_noise, label='High-Frequency Noise Removed', linestyle=':')
    plt.plot(final_ecg_signals, label='Final De-noised ECG signal')

    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with Noise Removal')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_noise_cancelling()
print(final_ecg_signal(ecg_signal))