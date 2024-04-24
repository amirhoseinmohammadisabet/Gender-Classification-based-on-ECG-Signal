import wfdb
import os

data_dir = 'mydata'
record_names = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')]

record = record_names[1]
signals, fields = wfdb.rdsamp(os.path.join(data_dir, record))

# print("Signals:", signals)
# print("Fields:", fields)

import matplotlib.pyplot as plt
import numpy as np

def plot_record(record_name, data_dir):
    signals, fields = wfdb.rdsamp(os.path.join(data_dir, record_name))
    fs = fields['fs']
    time = np.arange(len(signals)) / fs

    # Plot each signal
    # for i in range(len(signals[0])):
    #     plt.plot(time, signals[:, i], label='Signal {}'.format(i + 1))

    plt.plot(time, signals[:, 1], label='Signal {}')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Signals of Record: {}'.format(record_name))
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
# plot_record(record_names[0], data_dir)


import os
import pandas as pd
import wfdb

data_dir = 'mydata'
record_names = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')]

dataframes = []
column_names = []
for i in range(0, 7):
    record = record_names[i]
    signals, fields = wfdb.rdsamp(os.path.join(data_dir, record))
    df = pd.DataFrame(signals, columns=fields['sig_name'])
    dataframes.append((df.iloc[:, 0])[0:2000])
    column_names.append(f"ECG{i+1}")

combined_df = pd.concat(dataframes, axis=1) 
combined_df.columns = column_names

combined_df.to_csv('combined_ecg_data.csv', index=False)


