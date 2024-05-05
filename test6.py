import numpy as np
import pandas as pd
import wfdb

def extract_fft_features(signal, num_features=20):
    """Extract FFT features from a signal."""
    fft_result = np.fft.fft(signal)
    return np.abs(fft_result[:num_features])

def process_record(record_name):
    """Process a single record."""
    segment_features_list = []
    for segment_id in range(50):  # Assuming there are 50 segments per record
        segment_name = f'{record_name}_s{segment_id:02d}'
        try:
            record = wfdb.rdrecord(segment_name, channel_names=['ECG'])  
            signal = record.p_signal[:, 0]  # Assuming single lead ECG
            features = extract_fft_features(signal)
            segment_features_list.append(features)
        except Exception as e:
            print(f"Error processing segment {segment_name}: {e}")
    return segment_features_list

# Process all patient records
num_patients = 11
all_features_list = []
for patient_id in range(num_patients):
    patient_name = f'p{patient_id:02d}'
    for segment_features in process_record(patient_name):
        all_features_list.extend(segment_features)

# Convert to DataFrame
df_features = pd.DataFrame(all_features_list)

# Save features to CSV
df_features.to_csv("fft_features.csv", index=False)
