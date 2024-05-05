import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
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

# Load ECG data from CSV file
ecg_data = pd.read_csv('ECG_signal_test.csv')

# Extract features (e.g., mean, variance, etc.)
# You may need to adjust feature extraction based on your domain knowledge
# Here, I'm just using the first column as a placeholder for feature extraction
X = ecg_data.iloc[:, 0].values.reshape(-1, 1)
y = ecg_data.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_clf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train_scaled, y_train)
best_rf_clf = grid_search_rf.best_estimator_

# Hyperparameter tuning for AdaBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}
ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), random_state=42)
grid_search_ada = GridSearchCV(ada_clf, param_grid, cv=5, scoring='accuracy')
grid_search_ada.fit(X_train_scaled, y_train)
best_ada_clf = grid_search_ada.best_estimator_

# Ensemble method: Voting Classifier
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('random_forest', best_rf_clf),
    ('adaboost', best_ada_clf)
], voting='hard')

# Train and evaluate the models
for clf in (best_rf_clf, best_ada_clf, voting_clf):
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{clf.__class__.__name__}: Accuracy = {accuracy}")
