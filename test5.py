import scipy.io
import matplotlib.pyplot as plt
import pandas as pd

# Load .mat ECG data
mat_file_path = 'Data/JS00839.mat'
ecg_data = scipy.io.loadmat(mat_file_path)
x = ecg_data.values()
x2 = ecg_data['val']
# Print the keys in the loaded data
print("Keys in the loaded .mat file:", len(x2))

# Automatically select the first variable as the ECG signal
# You may need to adjust this if your .mat file has a different structure
variable_names = list(ecg_data.keys())
if variable_names:
    first_variable_name = variable_names[0]
    ecg_signal = ecg_data[first_variable_name]
for i in range(len(x2)):

    # Plot the ECG signal
    plt.plot(x2[i])
    plt.title(f'ECG Signal ({first_variable_name})')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()
else:
    print("No variables found in the loaded .mat file.")
