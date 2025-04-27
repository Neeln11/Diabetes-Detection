import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Display basic information
print("Dataset Overview:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Data Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Split data into features (X) and target (y)
X = df.drop("Outcome", axis=1)  # Keep all 8 features
y = df["Outcome"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM Model
svm_model = SVC(kernel='rbf', random_state=42)  # Using RBF kernel for SVM
svm_model.fit(X_train, y_train)

# Predictions
svm_pred = svm_model.predict(X_test)

# Accuracy Score
svm_accuracy = accuracy_score(y_test, svm_pred)

print("\nSVM Accuracy:", round(svm_accuracy * 100, 2), "%")

# Confusion Matrix & Classification Report for SVM
print("\nConfusion Matrix for SVM:")
print(confusion_matrix(y_test, svm_pred))

print("\nClassification Report for SVM:")
print(classification_report(y_test, svm_pred))

# Save the best model (SVM) using joblib
joblib.dump(svm_model, "diabetes_svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nSVM Model saved as 'diabetes_svm_model.pkl' and scaler as 'scaler.pkl'")
