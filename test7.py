import wfdb
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def file_to_csv(az,ta,long,to):
    data_dir = 'mydata'
    record_names = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')]

    dataframes = []
    column_names = []
    for i in range(az, ta):
        record = record_names[i]
        signals, fields = wfdb.rdsamp(os.path.join(data_dir, record))
        df = pd.DataFrame(signals, columns=fields['sig_name'])
        dataframes.append((df.iloc[:, 0])[0:long])
        column_names.append(f"ECG{i+1}")

    combined_df = pd.concat(dataframes, axis=1) 
    combined_df.columns = column_names
    combined_df.to_csv(f'{to}.csv', index=False)


# file_to_csv(0,10,20,"ecg_0-10")


def csv_merger():
    files = ["ecg-0-300.csv", "ecg-300-500.csv", "ecg-500-700.csv", "ecg-700-900.csv", "ecg-900-1120.csv"]
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file))
    merged_df = pd.concat(dfs, axis=1)
    merged_df.to_csv("ECG_signals_col.csv", )
    merged_df.T.to_csv("ECG_signals.csv", )
    print("Merged file saved successfully.")

# csv_merger()


def database_wrapper():
    df2 = pd.read_csv("ECG_signals.csv")
    df1 = pd.read_csv("subject-info.csv")
    merged_df = pd.merge(df2,df1,  left_index=True, right_index=True)
    merged_df.to_csv("ECG_signals_data.csv", )
    
    print(merged_df.head())

# database_wrapper()



def plot_record(signal_num, signal_len):
    fs = 1000
    df = pd.read_csv('ECG_signals_col.csv')
    ecg_signal = df.iloc[:,signal_num][0:signal_len]
    time = np.arange(len(ecg_signal))/fs

    plt.plot(time, ecg_signal, label='Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Signals of Record: {}'.format(signal_num))
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
plot_record(100,500)
