import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. Generate Highly Imbalanced Clinical Data (9900 Healthy, 100 Sick)
num_samples = 10000
features = 10

# Healthy patients (Class 0)
X_healthy = np.random.randn(9900, features)
y_healthy = np.zeros((9900, 1))

# Sick patients (Class 1) - Slightly different distribution
X_sick = np.random.randn(100, features) + 1.5 
y_sick = np.ones((100, 1))

# Combine and shuffle
X = np.vstack((X_healthy, X_sick))
y = np.vstack((y_healthy, y_sick))

# Convert to Tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 2. Define the Neural Network
class ClinicalNetwork(nn.Module):
    def __init__(self):
        super(ClinicalNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # We DO NOT put a sigmoid here because BCEWithLogitsLoss handles it better
        x = self.fc2(x)
        return x

# 3. Implement Custom Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Balances positive/negative importance
        self.gamma = gamma # Down-weights easy, confident predictions

    def forward(self, inputs, targets):
        # BCEWithLogitsLoss combines Sigmoid and Binary Cross Entropy for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate probabilities
        pt = torch.exp(-bce_loss) 
        
        # Apply the Focal Loss formula: -alpha * (1-pt)^gamma * log(pt)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return torch.mean(focal_loss)

# 4. Setup and Train
model = ClinicalNetwork()
# We use our new custom Focal Loss instead of standard BCELoss!
criterion = FocalLoss(alpha=0.75, gamma=2.0) 
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training model with Focal Loss on highly imbalanced data (99:1 ratio)...")
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

print("\nTraining complete. The model is now forced to pay attention to the minority class!")