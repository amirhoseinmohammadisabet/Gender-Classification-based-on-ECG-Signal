import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA


# Load ECG data from CSV file
ecg_data = pd.read_csv('ecg_0-10.csv').dropna()

# Extract features (FFT)
X = ecg_data.iloc[:, :-1]  # Exclude the last column (labels)
y = ecg_data.iloc[:, -1]   # Labels

# Perform FFT on each row
X_fft = np.abs(np.fft.fft(X, axis=1))

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fft)

# Impute missing values
imputer = SimpleImputer(strategy='mean')  # You can change the strategy if needed
X_imputed = imputer.fit_transform(X_scaled)

# PCA for dimensionality reduction
pca = PCA(n_components=5)  # Choose appropriate number of components
X_pca = pca.fit_transform(X_imputed)

# Feature Selection
selector = SelectKBest(f_classif, k=5)  # Select top 5 features
X_selected = selector.fit_transform(X_pca, y)

# Split data into training and testing sets for selected features
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "ANN": MLPClassifier()
}

# Train classifiers with selected features
trained_classifiers = {}
for name, clf in classifiers.items():
    clf.fit(X_train_sel, y_train_sel)
    trained_classifiers[name] = clf

# Evaluate classifiers
results = {}
confusion_matrices = {}
for name, clf in trained_classifiers.items():
    y_pred = clf.predict(X_test_sel)
    accuracy = accuracy_score(y_test_sel, y_pred)
    confusion_matrices[name] = confusion_matrix(y_test_sel, y_pred)
    results[name] = accuracy

# Print results
for name, accuracy in results.items():
    print(f"{name}: Accuracy = {accuracy}")

# Plot confusion matrices
plt.figure(figsize=(15, 10))
for i, (name, matrix) in enumerate(confusion_matrices.items(), 1):
    plt.subplot(2, 3, i)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
plt.tight_layout()
plt.show()
