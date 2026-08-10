import torch
import torch.nn as nn
import torch.optim as optim

# 1. Input Data and Ground Truth (Notice they are Tensors now)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# 2. Define the Neural Network Architecture
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # PyTorch creates the weights and biases for us!
        self.hidden_layer = nn.Linear(in_features=2, out_features=2)
        self.output_layer = nn.Linear(in_features=2, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # The forward pass is just passing data through the layers
        hidden_out = self.sigmoid(self.hidden_layer(x))
        final_out = self.sigmoid(self.output_layer(hidden_out))
        return final_out

# 3. Instantiate the Model, Loss, and Optimizer
model = SimpleMLP()
criterion = nn.MSELoss() # Mean Squared Error (calculates the blame)
optimizer = optim.SGD(model.parameters(), lr=0.5) # Stochastic Gradient Descent

# 4. The Training Loop
epochs = 10000
for epoch in range(epochs):
    # Step A: Forward Pass (Make a guess)
    predicted_output = model(X)
    
    # Step B: Calculate Loss (How bad was the guess?)
    loss = criterion(predicted_output, y)
    
    # Step C: Backpropagation
    optimizer.zero_grad() # Clear the old tape recorder
    loss.backward()       # Play the tape backward (do the calculus)
    optimizer.step()      # Update the weights (learn!)
    
    # Print progress every 2000 epochs
    if epoch % 2000 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

print("\nFinal Predictions after training:")
print(model(X).detach())