import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/Data/The_Cancer_data_1500_V2.csv")

# Replace Gender codes with labels for better understanding
# df['Gender'] = df['Gender'].replace({0: 'Male', 1: 'Female'})

# Step 1: Column Information and Data Types
print("Column Information and Data Types:")
print(df.info())

# Step 3: Missing Values
print("\nMissing Values Count:")
print(df.isnull().sum())

# Save EDA text report
with open("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/Data/EDA_Report.txt", "w") as file:
    file.write("Column Information and Data Types:\n")
    file.write(str(df.dtypes) + "\n\n")
    file.write("Summary Statistics:\n")
    file.write(df.describe().to_string() + "\n\n")
    file.write("Missing Values Count:\n")
    file.write(df.isnull().sum().to_string() + "\n\n")

# Step 4: Visualization - Numerical Feature Distributions
numerical_columns = ['Age','BMI', 'PhysicalActivity', 'AlcoholIntake']

plt.figure(figsize=(12, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[column], kde=True)
    plt.title(f'{column} Distribution')
plt.tight_layout()
plt.savefig("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/Data/The_Cancer_data_1500_V2.csv")
plt.show()

# Step 5: Boxplots (Outlier Detection)
plt.figure(figsize=(12, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df[column])
    plt.title(f'{column} Boxplot')
plt.tight_layout()
plt.savefig("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/Data/eda_boxplots.png")
plt.show()

# Step 6: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/Data/eda_correlation_heatmap.png")
plt.show()

print("\n EDA Completed - Reports and Visualizations Saved!")

# Create processed dataset and save
df.to_csv("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/Data/The_Cancer_data_1500_V2_Processed.csv", index=False)