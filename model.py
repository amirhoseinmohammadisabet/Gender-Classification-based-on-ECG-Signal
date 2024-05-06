import pandas as pd
import numpy as np
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns


# Load ECG data from CSV file
try:
    ecg_data = pd.read_csv('ECG_signal_main.csv').dropna()
except FileNotFoundError:
    print("Error: File 'ECG_signal_main' not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: The file 'ECG_signal_main' is empty.")
    exit(1)
except pd.errors.ParserError:
    print("Error: Unable to parse the CSV file 'ECG_signal_main'. Please check the file format.")
    exit(1)

# Check if the dataset has enough records
if len(ecg_data) < 20:
    print("Error: The dataset has insufficient records. At least 20 records are required.")
    exit(1)


# Extract features (FFT)
X = ecg_data.iloc[:, :-1]  # Exclude the last column (labels)
y = ecg_data.iloc[:, -1]   # Labels

# Perform FFT on each row
X_fft = np.abs(fft(X, axis=1))

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
selector = SelectKBest(f_classif, k=5)  
X_selected = selector.fit_transform(X_pca, y)

# Split data into training and testing sets for selected features
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Define classifiers with hyperparameter tuning
classifiers = {
    "Decision Tree": GridSearchCV(DecisionTreeClassifier(), {'max_depth': [None, 10, 20]}),
    "Random Forest": GridSearchCV(RandomForestClassifier(), {'n_estimators': [100, 200, 300]}),
    "SVM": GridSearchCV(SVC(), {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
    "Naive Bayes": GaussianNB(),
    "KNN": GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    "ANN": GridSearchCV(MLPClassifier(), {'hidden_layer_sizes': [(100,), (50, 50), (100, 50)]})
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


# Plot histograms for selected features
plt.figure(figsize=(10, 6))
for i in range(X_selected.shape[1]):
    plt.subplot(2, 3, i+1)
    plt.hist(X_selected[:, i], bins=30, edgecolor='black')
    plt.title(f'Feature {i+1} Distribution')
plt.tight_layout()
plt.show()



# Create a pairplot for selected features
selected_features_df = pd.DataFrame(X_selected, columns=[f'Feature_{i+1}' for i in range(X_selected.shape[1])])
selected_features_df['Target'] = y
sns.pairplot(selected_features_df, hue='Target')
plt.show()


# Visualize PCA components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Components')
plt.colorbar(label='Target')
plt.show()

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

