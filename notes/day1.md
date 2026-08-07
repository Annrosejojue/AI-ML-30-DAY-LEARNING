The Input Layer (The Rookies): These detectives just look at the raw clues (blood pressure, age, heart rate). They don't do any thinking; they just pass the clues to the next group.

The Hidden Layers (The Specialists): These detectives receive the clues. They have specific biases (baseline suspicions) and assign weights (importance) to different clues. They combine everything they receive, apply a threshold (an activation function) to decide if the combination means anything, and pass their findings forward.

The Output Layer (The Chief): The final detective takes all the specialists' reports and makes the final diagnosis (e.g., "90% chance of high risk"). This entire process from Rookies to the Chief is called Forward Propagation

Neuron / Node: The basic computational unit. It computes a linear combination of its inputs (Weights * Inputs + Bias) and passes the result through a non-linear function.
Weights (W): Matrices that dictate the strength of the connection between a neuron in one layer and a neuron in the next. They are the parameters the network actually "learns."
Bias (b): A constant added to the weighted sum before the activation function, allowing the activation curve to shift left or right.
Activation Function: A mathematical equation (like Sigmoid or ReLU) applied to a neuron's output to introduce non-linearity, allowing the network to learn complex patterns instead of just straight lines.
Forward Pass: The sequential calculation of outputs from the input layer to the output layer.
Loss Function (L): A mathematical metric (like Mean Squared Error or Cross-Entropy) evaluating the difference between the model's prediction and the actual ground truth.
Backpropagation: An algorithm that utilizes the chain rule of calculus to compute the gradient (slope) of the loss function with respect to every single weight and bias in the network.  Gradient Descent: The optimization algorithm that actually updates the weights by taking a small step in the opposite direction of the gradient to minimize the loss.  

