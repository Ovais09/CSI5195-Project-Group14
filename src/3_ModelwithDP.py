import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the Dataset
df = pd.read_csv("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/Data/The_Cancer_data_1500_V2_Processed.csv")

# Step 3: Apply Differential Privacy (DP)
def add_gaussian_noise(data, mean=0, std=0.5):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

# Apply noise to sensitive features
sensitive_features = ["BMI", "PhysicalActivity", "AlcoholIntake"]
df[sensitive_features] = add_gaussian_noise(df[sensitive_features])

# Save the new noised dataset
df.to_csv("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/Data/The_Cancer_data_1500_V2_Noised.csv", index=False)
print("\n Modified dataset with noise saved as: The_Cancer_data_1500_V2_Noised.csv")
