1. Normalization (Leveling the Playing Field)
Imagine a patient's record. Their "Pain Level" is a number between 1 and 10. Their "White Blood Cell Count" is a number around 10,000.

If you feed these raw numbers into the neural network, the Chief Detective (the output layer) gets incredibly confused. It thinks the White Blood Cell count is 1,000 times more important than the Pain Level simply because the number is bigger. It's like one detective is whispering and the other is screaming into a megaphone.

Normalization (or Scaling) uses basic math to compress every single column of data so that they all fit into the exact same range (usually between -3 and 3). Now, every detective speaks at the exact same volume, and the network can figure out which clue is actually the most important.

Tabular Data: Data structured into a table with rows (individual patients) and columns (features).

Feature Scaling / Standardization: Transforming data to have a mean of 0 and a standard deviation of 1.

Overfitting: When a model learns the training data too well, memorizing the noise and failing to generalize to new, unseen data.

Train/Test Split: Dividing a dataset into a training set (used to adjust weights) and a testing set (used exclusively to evaluate performance).

torch.no_grad(): A PyTorch command used during testing. It turns off the "Autograd tape recorder" so the model doesn't accidentally learn from the test exam, and it saves computer memory.