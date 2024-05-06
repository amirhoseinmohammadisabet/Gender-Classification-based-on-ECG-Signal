import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def ecgin(pernum, lengsig):
    ecgsig = np.zeros((pernum, lengsig))
    data_folder = "Data/MIT_BIH_NS"  # Specify the folder name
    for x in range(1, pernum+1):
        filename = f"{x}.mat"
        filepath = os.path.join(data_folder, filename)
        data = sio.loadmat(filepath)
        ecgsig = data['val']
        # You can add filtering or denoising steps here if needed
    return ecgsig


def plot_ecg(ecgsig, sample_rate=1):
    num_ecgs, num_samples = ecgsig.shape
    print(num_ecgs)
    print(num_samples)
    time = np.arange(0, num_samples/sample_rate, 1/sample_rate)
    
    plt.figure(figsize=(10, 6))
    for i in range(num_ecgs):
        plt.plot(time, ecgsig[i], label=f'ECG {i+1}')

    plt.title('ECG Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define the parameters
pernum = 2
lengsig = 21000

# Call the function
ecgsig = ecgin(pernum, lengsig)
print(ecgsig)
plot_ecg(ecgsig, sample_rate=2)