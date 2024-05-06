import pandas as pd
import numpy as np
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif

# Load ECG data from CSV file
ecg_data = pd.read_csv('ECG_signal_main.csv')
ecg_data = ecg_data.dropna()

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
selector = SelectKBest(f_classif, k=5)  # Select top 5 features
X_selected = selector.fit_transform(X_pca, y)

# Define classifiers with hyperparameter tuning
classifiers = {
    "Decision Tree": GridSearchCV(DecisionTreeClassifier(), {'max_depth': [None, 10, 20]}, cv=5),
    "Random Forest": GridSearchCV(RandomForestClassifier(), {'n_estimators': [100, 200, 300]}, cv=5),
    "SVM": GridSearchCV(SVC(), {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}, cv=5),
    "Naive Bayes": GaussianNB(),
    "KNN": GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}, cv=5),
    "ANN": GridSearchCV(MLPClassifier(), {'hidden_layer_sizes': [(100,), (100, 100), (100, 50)]}, cv=5)
}

# Train and evaluate classifiers with selected features using k-fold cross-validation
results = {}
for name, clf in classifiers.items():
    if name == "Naive Bayes":
        scores = cross_val_score(clf, X_selected, y, cv=5)
        accuracy = np.mean(scores)
    else:
        clf.fit(X_selected, y)
        cv_results = clf.cv_results_
        accuracy = np.mean(cv_results['mean_test_score'])
    results[name] = accuracy

# Print results
for name, accuracy in results.items():
    print(f"{name}: Accuracy = {accuracy}")
