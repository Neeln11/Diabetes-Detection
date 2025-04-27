# train_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Use all features (no column dropping)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Handle imbalance
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM hyperparameters
svm_params = {
    "C": [0.1, 1, 10, 50, 100, 200, 500],
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "gamma": [0.001, 0.01, 0.1, 1, "scale", "auto"],
    "class_weight": [None, "balanced"]
}

# Train model using GridSearchCV
svm_grid = GridSearchCV(SVC(), svm_params, cv=10, scoring="accuracy", n_jobs=-1, verbose=2)
svm_grid.fit(X_train, y_train)

# Best model
best_svm = svm_grid.best_estimator_

# Evaluate
svm_accuracy = accuracy_score(y_test, best_svm.predict(X_test))
print(f"\n🔹 Optimized SVM Accuracy with 8 features: {svm_accuracy * 100:.2f}%")

# Save model and scaler
joblib.dump(best_svm, "best_svm_model_8features.pkl")
joblib.dump(scaler, "scaler_8features.pkl")

print("\n✅ Model saved as 'best_svm_model_8features.pkl' and scaler as 'scaler_8features.pkl'")
