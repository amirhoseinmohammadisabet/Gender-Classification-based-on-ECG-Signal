import numpy as np
import pandas as pd
import wfdb

data_path = "your_data_path"  # Update with your actual data path

def extract_fft_features(signal, num_features=20):
    """Extract FFT features from a signal."""
    fft_result = np.fft.fft(signal)
    return np.abs(fft_result[:num_features])

def process_patient(patient_id, data_path):
    """Process a single patient."""
    features = []
    for segment_id in range(1):  # Only considering segment 0
        filename = f'{data_path}/p0{str(patient_id)[:1]}/p{patient_id:05d}/p{patient_id:05d}_s{segment_id:02d}'
        rec = wfdb.rdrecord(filename)
        signal = rec.p_signal[:, 0]  # Assuming single lead ECG
        features.append(extract_fft_features(signal))
    return np.array(features)

# Process first 2000 patients
num_patients = 2000
features_list = []
for patient_id in range(1, num_patients + 1):
    features = process_patient(patient_id, data_path)
    features_list.append(features)

# Convert to DataFrame
df_features = pd.DataFrame(np.vstack(features_list))

# Save features to CSV
df_features.to_csv("fft_features.csv", index=False)
