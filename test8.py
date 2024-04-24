
import wfdb
0001.dat
patient_id=1
segment_id=00
start=20000
length=1024
filename = f'{"mydata"}/{patient_id:0d}'
print(filename)
rec = wfdb.rdrecord(filename, sampfrom=start, sampto=start+length)
ann = wfdb.rdann(filename, "atr", sampfrom=start, sampto=start+length, shift_samps=True)
wfdb.plot_wfdb(rec, ann, plot_sym=True, figsize=(15,4))



def x():
    import wfdb

    # Load the ECG record
    record_name = 'mydata/p01000_s00'
    record = wfdb.rdrecord(record_name)

    # Print some information about the record
    print(record.__dict__)

    # You can access the ECG signal data
    ecg_signal = record.p_signal

    # If you want to plot the ECG signal
    import matplotlib.pyplot as plt
    plt.plot(ecg_signal)
    plt.xlabel('Sample')
    plt.ylabel('ECG Signal')
    plt.title('ECG Signal')
    plt.show()