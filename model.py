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
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif

# Load ECG data from CSV file
ecg_data = pd.read_csv('ecg_0-10.csv')
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

# Train and evaluate classifiers with selected features
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train_sel, y_train_sel)
    y_pred = clf.predict(X_test_sel)
    accuracy = accuracy_score(y_test_sel, y_pred)
    results[name] = accuracy

# Print results
for name, accuracy in results.items():
    print(f"{name}: Accuracy = {accuracy}")
