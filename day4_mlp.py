import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Load and Prepare the Clinical Data
data = load_breast_cancer()
X = data.data    # The 30 clinical features (tumor radius, texture, etc.)
y = data.target  # 0 = Malignant, 1 = Benign

# The Vault: Split into 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization: Leveling the playing field
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Learn the scale from training data
X_test_scaled = scaler.transform(X_test)       # Apply that same scale to the test data

# Convert NumPy arrays to PyTorch Tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Package into DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. Define the Neural Network Architecture
class ClinicalEHRNetwork(nn.Module):
    def __init__(self):
        super(ClinicalEHRNetwork, self).__init__()
        # 30 input features based on the breast cancer dataset
        self.layer1 = nn.Linear(30, 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# 3. Setup Model, Loss, and Optimizer
model = ClinicalEHRNetwork()
criterion = nn.BCELoss() # Binary Cross Entropy for 0 or 1 classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training Loop
epochs = 50
print("Training on 80% of patient records...")
for epoch in range(epochs):
    model.train() # Put model in training mode
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

print("Training Complete.\n")

# 5. Testing Loop (The Final Exam)
print("Evaluating on 20% unseen patient records...")
model.eval() # Put model in evaluation mode (turns off some training-specific behaviors)

# Turn off the Autograd tape recorder! We don't want to learn from the test set.
with torch.no_grad(): 
    test_predictions = model(X_test_t)
    # Convert probabilities (>0.5) to binary 0 or 1
    predicted_classes = (test_predictions >= 0.5).float()
    
    # Compare predictions to the actual answers
    correct = (predicted_classes == y_test_t).sum().item()
    total = y_test_t.size(0)
    accuracy = (correct / total) * 100

print(f"Test Accuracy: {accuracy:.2f}%")