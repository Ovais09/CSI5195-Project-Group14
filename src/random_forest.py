import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load dataset (replace 'your_dataset.csv' with actual file)
df = pd.read_csv("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/Data/The_Cancer_data_1500_V2_Processed.csv")

# Define feature columns and target variable
feature_cols = ["Age", "Gender", "BMI", "Smoking", "GeneticRisk", "PhysicalActivity", "AlcoholIntake", "CancerHistory"]
target_col = "Diagnosis"  # Change this if the column name is different

X = df[feature_cols]
y = df[target_col]

# Convert categorical features to numerical values
X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1})
X["Smoking"] = X["Smoking"].map({"No": 0, "Yes": 1})
X["GeneticRisk"] = X["GeneticRisk"].map({"Low": 0, "Medium": 1, "High": 2})
X["CancerHistory"] = X["CancerHistory"].map({"No": 0, "Yes": 1})

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier (without DP)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"🔹 Accuracy: {accuracy:.2f}")
print(f"🔹 ROC-AUC Score: {roc_auc:.2f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="blue", label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest Model")
plt.legend()
plt.show()