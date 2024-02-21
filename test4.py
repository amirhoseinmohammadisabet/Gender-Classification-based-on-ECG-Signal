import scipy.io
import matplotlib.pyplot as plt

# Load .mat ECG data
mat_file_path = 'Data/JS45500.mat'
ecg_data = scipy.io.loadmat(mat_file_path)

# Print the keys in the loaded data
print("Keys in the loaded .mat file:", ecg_data.keys())

# Automatically select the first variable as the ECG signal
# You may need to adjust this if your .mat file has a different structure
variable_names = list(ecg_data.keys())
if variable_names:
    first_variable_name = variable_names[0]
    ecg_signal = ecg_data[first_variable_name]

    # Plot the ECG signal
    plt.plot(ecg_signal)
    plt.title(f'ECG Signal ({first_variable_name})')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()
else:
    print("No variables found in the loaded .mat file.")
