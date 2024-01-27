import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('ecg.csv', header=None)

# Assuming the last column is the label (1 for abnormal, 0 for normal)
labels = df.iloc[:, -1]
ecg_data = df.iloc[:, :-1]

# Function to compute Fourier transform features
def compute_fourier_features(data):
    # Apply Fourier transform to each row of the data
    fft_result = np.fft.fft(data, axis=1)
    
    # Take the magnitude of the Fourier transform
    magnitude = np.abs(fft_result)
    
    # Use the first half of the magnitudes as features
    features = magnitude[:, :len(data.columns)//2]
    
    return features

# Compute Fourier transform features
fourier_features = compute_fourier_features(ecg_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(fourier_features, labels, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', C=1)

# Perform cross-validation
cv_scores = cross_val_score(svm_classifier, X_train, y_train, cv=5, scoring='accuracy')

# Train the model on the training set
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Accuracy:", np.mean(cv_scores))
print("Test Set Accuracy:", accuracy)

# Optional: Plot an example ECG signal and its Fourier transform
example_row = ecg_data.iloc[0]
example_label = labels.iloc[0]
example_fft = np.fft.fft(example_row)
example_magnitude = np.abs(example_fft)[:len(example_row)//2]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(example_row)
plt.title(f'Example ECG Signal (Label: {example_label})')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(example_magnitude)
plt.title('Fourier Transform')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
