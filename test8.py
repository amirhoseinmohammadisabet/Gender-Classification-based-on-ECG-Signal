import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


# Function to compute Fourier transform features
def compute_fourier_features(data):
    fft_result = np.fft.fft(data, axis=1)
    magnitude = np.abs(fft_result)
    features = magnitude[:, :30]
    return features

# Function to perform ECG classification
def ecg_classification(X_train, X_test, y_train):
    # Initialize SVM classifier
    svm_classifier = SVC(kernel='linear', C=1)

    # Perform cross-validation
    cv_scores = cross_val_score(svm_classifier, X_train, y_train, cv=10, scoring='accuracy')

    # Train the model on the training set
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test)

    return cv_scores, y_pred


def ecg_classification_rf(X_train, X_test, y_train):
    # Initialize Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=10, scoring='accuracy')

    # Train the model on the training set
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    return cv_scores, y_pred

def ecg_classification_dt(X_train, X_test, y_train):
    # Initialize Decision Tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(dt_classifier, X_train, y_train, cv=10, scoring='accuracy')

    # Train the model on the training set
    dt_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = dt_classifier.predict(X_test)

    return cv_scores, y_pred

# Function to evaluate and visualize results
def evaluate_and_visualize(cv_scores, y_test, y_pred, X_test_pca):
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display results
    print("Cross-Validation Scores:", cv_scores)
    print("Average Cross-Validation Accuracy:", np.mean(cv_scores))
    print("Test Set Accuracy:", accuracy)

    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Visualize scatter plot for correct and incorrect predictions
    correct_predictions = X_test_pca[y_test == y_pred]
    incorrect_predictions = X_test_pca[y_test != y_pred]

    plt.figure(figsize=(12, 8))
    plt.scatter(correct_predictions[:, 0], correct_predictions[:, 1], c='green', label='Correct Prediction')
    plt.scatter(incorrect_predictions[:, 0], incorrect_predictions[:, 1], c='red', label='Incorrect Prediction')
    plt.title('Scatter Plot of Correct and Incorrect Predictions')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

file_path='ECG_signals_BMI.csv'
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path, header=None)
df = df.dropna()

# Assuming the last column is the label (1 for abnormal, 0 for normal)
labels = df.iloc[:, -1]
ecg_data = df.iloc[:, :-1]

# Compute Fourier transform features
fourier_features = compute_fourier_features(ecg_data)
ff = pd.DataFrame(fourier_features)
print(ff.head())
ff.to_csv("fourier_features.csv")


# Apply PCA to Fourier features
fourier_features = pd.read_csv("fourier_features.csv")
pca = PCA(n_components=10)
pca_features = pca.fit_transform(ecg_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(pca_features, labels, test_size=0.2, random_state=100)

# Perform ECG classification
cv_scores, y_pred = ecg_classification_rf(X_train, X_test, y_train)

# Evaluate and visualize results
evaluate_and_visualize(cv_scores, y_test, y_pred, X_test)
