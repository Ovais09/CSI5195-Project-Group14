import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/Data/The_Cancer_data_1500_V2_Noised.csv")

# Split Features & Target Variable
X = df.drop(columns=["Diagnosis"]).values  # Convert to NumPy array
y = df["Diagnosis"].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for DP-SGD
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define Logistic Regression Model in PyTorch
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 2)  # 2 outputs (Cancer Positive/Negative)
    
    def forward(self, x):
        return self.linear(x)

model = LogisticRegressionModel(input_size=X_train.shape[1])

# Define Loss Function & Optimizer (DP-SGD)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Apply Differential Privacy Using Opacus
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=10,
    target_epsilon=3.0,  # Privacy Budget
    target_delta=1e-5,
    max_grad_norm=1.0,  # Clipping Gradient Norm
)

# Train the DP-SGD Model
for epoch in range(10):
    for batch_idx, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

# Evaluate Model on Test Data
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor).argmax(dim=1).numpy()

# Calculate Accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\n DP-SGD Test Accuracy: {test_accuracy:.4f}")

# Save Model & Scaler
torch.save(model.state_dict(), "C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/model/dp_sgd_model.pth")
torch.save(scaler, "C:/Users/sugan/Desktop/DP-SGD/CSI5195-Project-Group14/model/scaler_SGD.pkl")
print("\n DP-SGD Model & Scaler Saved Successfully!")