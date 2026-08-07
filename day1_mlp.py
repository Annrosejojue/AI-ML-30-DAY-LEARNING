import numpy as np

# 1. Activation Function & its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Input Data (X) and Ground Truth (y) - The XOR Problem
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 3. Initialize Random Weights and Biases
np.random.seed(42)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Weights
W_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
W_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Biases
b_hidden = np.random.uniform(size=(1, hidden_neurons))
b_output = np.random.uniform(size=(1, output_neurons))

learning_rate = 0.5
epochs = 10000

# 4. The Training Loop
for epoch in range(epochs):
    # --- FORWARD PROPAGATION ---
    hidden_layer_activation = np.dot(X, W_hidden) + b_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, W_output) + b_output
    predicted_output = sigmoid(output_layer_activation)
    
    # --- CALCULATE LOSS (Error) ---
    error = y - predicted_output
    
    # --- BACKPROPAGATION ---
    # How much did the output layer miss by?
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # How much did the hidden layer contribute to the output error?
    error_hidden_layer = d_predicted_output.dot(W_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # --- GRADIENT DESCENT (Update Weights & Biases) ---
    W_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    b_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    
    W_hidden += X.T.dot(d_hidden_layer) * learning_rate
    b_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

print("Final Predictions after training:")
print(predicted_output)