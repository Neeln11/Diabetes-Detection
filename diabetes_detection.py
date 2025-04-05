import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Display dataset overview
print("Dataset Overview:\n", df.head())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Remove less important features (based on correlation)
df = df.drop(columns=['SkinThickness', 'DiabetesPedigreeFunction'])  # Less important features

# Split data into features (X) and target (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optimized SVM Hyperparameters
svm_params = {
    "C": [0.1, 1, 10, 50, 100, 200, 500],  # More variations in C
    "kernel": ["linear", "rbf", "poly", "sigmoid"],  # Adding sigmoid kernel
    "gamma": [0.001, 0.01, 0.1, 1, "scale", "auto"],  # Fine-tuning gamma
    "class_weight": [None, "balanced"]  # Handling class imbalance
}

# Train using GridSearchCV with more cross-validation folds
svm_grid = GridSearchCV(SVC(), svm_params, cv=10, scoring="accuracy", n_jobs=-1, verbose=2)
svm_grid.fit(X_train, y_train)

# Best SVM Model
best_svm = svm_grid.best_estimator_
svm_accuracy = accuracy_score(y_test, best_svm.predict(X_test))

# Print accuracy results
print("\n🔹 Model Accuracy:")
print(f"Optimized SVM Accuracy: {svm_accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, best_svm.predict(X_test)))

# Print confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, best_svm.predict(X_test)), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save the best model
joblib.dump(best_svm, "best_diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Best model saved as 'best_diabetes_model.pkl' with accuracy:", round(svm_accuracy * 100, 2), "%")
