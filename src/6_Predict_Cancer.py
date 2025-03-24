import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

# Define the model architecture
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 2)  # 2 output classes (Cancer Positive/Negative)

    def forward(self, x):
        return self.linear(x)

# Load DP-SGD Model using PyTorch
input_size = 8  # Adjust based on number of features
model = LogisticRegressionModel(input_size)

# Load the state dictionary and fix Opacus' _module. prefix issue
state_dict = torch.load("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/model/dp_sgd_model.pth")
new_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# Load Scaler using joblib
scaler = joblib.load("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/model/scaler_DP.pkl")

# Feature Names
feature_names = ["Age", "Gender", "BMI", "Smoking", "Genetic Risk", "Physical Activity", "Alcohol Intake", "Family Cancer History"]

# Function to take user input
def get_user_input():
    print("\nEnter patient details to predict cancer diagnosis:\n")

    age = float(input("Age (18-100): "))
    gender = input("Gender (Male/Female): ").strip().lower()
    bmi = float(input("BMI (10-50): "))
    smoking = input("Smoking Status (Yes/No): ").strip().lower()
    genetic_risk = input("Genetic Risk (Low/Medium/High): ").strip().lower()
    physical_activity = float(input("Physical Activity (hours per week, 0-50): "))
    alcohol_intake = float(input("Alcohol Intake (times per week, 0-20): "))
    cancer_history = input("Family History of Cancer (Yes/No): ").strip().lower()

    # Convert categorical inputs to numerical values
    gender = 1 if gender == "female" else 0
    smoking = 1 if smoking == "yes" else 0
    genetic_risk = {"low": 0, "medium": 1, "high": 2}[genetic_risk]
    cancer_history = 1 if cancer_history == "yes" else 0

    return np.array([[age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history]])

# Function to explain the prediction
def explain_prediction(user_input_scaled):
    # Compute decision scores before applying softmax
    decision_scores = model(user_input_scaled).detach().numpy()[0]

    # Get feature contributions (importance * input value)
    coefficients = model.linear.weight[1].detach().numpy()  # Get coefficients for "Cancer Positive" class
    feature_contributions = user_input_scaled[0].numpy() * coefficients

    # Sort features by absolute contribution
    sorted_features = sorted(zip(feature_names, feature_contributions), key=lambda x: abs(x[1]), reverse=True)

    # Convert to human-readable messages
    explanation = []
    for feature, contribution in sorted_features[:3]:  # Show top 3 reasons
        influence = "Increases Risk" if contribution > 0 else "Decreases Risk"
        if feature == "BMI":
            msg = "High BMI is linked to a greater risk of cancer."
        elif feature == "Smoking":
            msg = "Smoking increases the chance of developing cancer."
        elif feature == "Genetic Risk":
            msg = "A strong family history of cancer increases susceptibility."
        elif feature == "Physical Activity":
            msg = "Regular exercise helps reduce cancer risk."
        elif feature == "Alcohol Intake":
            msg = "Frequent alcohol consumption increases cancer risk."
        elif feature == "Age":
            msg = "Older age is generally associated with higher cancer risk."
        elif feature == "Gender":
            msg = "Some cancers have different risks based on gender."
        elif feature == "Family Cancer History":
            msg = "Having close relatives with cancer increases personal risk."
        else:
            msg = f"{feature} impacts cancer risk."

        explanation.append([feature, round(contribution, 3), influence, msg])

    # Print Explanation in Tabular Format
    print("\n Reasons for Prediction:\n")
    df_explanation = pd.DataFrame(explanation, columns=["Factor", "Impact", "Effect", "Explanation"])
    print(df_explanation.to_string(index=False))

    # Print Explanation of Impact
    print("\n What Does 'Impact' Mean?")
    print("- The higher the impact, the stronger the effect of this factor on the prediction.")
    print("- Positive impact values *increase cancer risk*.")
    print("- Negative impact values *reduce cancer risk*.")

# Function to predict cancer
def predict_cancer():
    user_input = get_user_input()

    # Scale input data using the saved scaler
    user_input_scaled = torch.tensor(scaler.transform(user_input), dtype=torch.float32)

    # Make prediction
    output = model(user_input_scaled)
    prediction = output.argmax().item()  # Get class index

    # Print result
    result = "Cancer Positive" if prediction == 1 else "Cancer Negative"
    print(f"\n Prediction: {result}")

    # Explain why the model made this prediction
    explain_prediction(user_input_scaled)

# Run the form
if __name__ == "__main__":
    predict_cancer()
