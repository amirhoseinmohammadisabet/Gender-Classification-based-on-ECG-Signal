import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve

# Read the CSV file into a DataFrame
df = pd.read_csv('ecg.csv', header=None)

# Assuming the last column is the label (1 for abnormal, 0 for normal)
labels = df.iloc[:, -1]
ecg_data = df.iloc[:, :-1]

# Function to compute Fourier transform features
def compute_fourier_features(data):
    fft_result = np.fft.fft(data, axis=1)
    magnitude = np.abs(fft_result)
    features = magnitude[:, :10]  # Keep the first 10 components
    return features

# Compute Fourier transform features
fourier_features = compute_fourier_features(ecg_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(fourier_features, labels, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', C=1)

# Train the model on the training set
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Function to plot ROC curve
def plot_roc_curve():
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plotting ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()

    st.pyplot(fig)


# Function to plot precision-recall curve
def plot_precision_recall_curve():
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    st.plotly_chart({
        "data": [{"x": recall, "y": precision, "type": "line"}],
        "layout": {"title": "Precision-Recall Curve", "xaxis": {"title": "Recall"}, "yaxis": {"title": "Precision"}}
    })

# Function to plot confusion matrix heatmap
def plot_confusion_matrix():
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.write("Confusion Matrix:")
    st.write(conf_matrix)

    st.plotly_chart({
        "data": [{"z": conf_matrix, "type": "heatmap", "colorscale": "Viridis"}],
        "layout": {"title": "Confusion Matrix"}
    })

# Function to visualize misclassified instances
def visualize_misclassifications():
    misclassifications = X_test[y_test != y_pred]

    st.scatter_chart(misclassifications, color='red', title='Misclassified Instances')

# Function to visualize decision boundaries (2D data only)
def visualize_decision_boundaries():
    if X_train.shape[1] == 2:
        # Extracting two features for visualization
        features_for_visualization = X_train.iloc[:, :2].values

        # Creating a meshgrid for decision boundary plotting
        h = .02  # Step size in the mesh
        x_min, x_max = features_for_visualization[:, 0].min() - 1, features_for_visualization[:, 0].max() + 1
        y_min, y_max = features_for_visualization[:, 1].min() - 1, features_for_visualization[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Plotting decision boundaries
        Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        st.plotly_chart({
            "data": [
                {"x": features_for_visualization[:, 0], "y": features_for_visualization[:, 1], "mode": "markers", "marker": {"color": y_train, "size": 8}},
                {"x": np.ravel(xx), "y": np.ravel(yy), "z": Z, "type": "contour", "colorscale": "Viridis"}
            ],
            "layout": {"title": "Decision Boundaries"}
        })

# Main Streamlit app
def main():
    st.title("ECG Classification Results")

    # Add other components and text to your app as needed

    # Visualizations
    st.header("Visualizations")

    st.subheader("ROC Curve")
    plot_roc_curve()

    st.subheader("Precision-Recall Curve")
    plot_precision_recall_curve()

    st.subheader("Confusion Matrix")
    plot_confusion_matrix()

    st.subheader("Misclassified Instances")
    visualize_misclassifications()

    st.subheader("Decision Boundaries")
    visualize_decision_boundaries()

if __name__ == "__main__":
    main()
