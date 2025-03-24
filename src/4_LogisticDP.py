import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/Data/The_Cancer_data_1500_V2_Noised.csv")

# Splitting dataset into features and target variable
X = df.drop(columns=["Diagnosis"]) #X is independent variable
y = df["Diagnosis"]  #Y is dependent variable

# Train-Test Split (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training (Logistic Regression)
model = LogisticRegression(max_iter=1000)  # increase max_iter if convergence warning happens
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/model/logistic_model.pkl")
joblib.dump(scaler, "C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/model/scaler.pkl")

# Model Evaluation
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Plot Accuracy Graph
plt.figure(figsize=(6, 4))
plt.bar(["Train Accuracy", "Test Accuracy"], [train_accuracy, test_accuracy], color=['blue', 'orange'])
plt.ylim(0, 1)
plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.show()

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion_matrix(y_test, y_test_pred)
