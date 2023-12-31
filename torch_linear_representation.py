import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import sin, cos

# Define the linear function
def linear_function(x):
    return sin(x)

# Generate training data
x_train = np.linspace(-10, 10, 100, dtype=np.float32)
y_train = linear_function(x_train)
# normalize
x_train = x_train / 10

# Convert numpy arrays to torch tensors
x_train = torch.from_numpy(x_train.reshape(-1, 1))
y_train = torch.from_numpy(y_train.reshape(-1, 1))

# Define the neural network model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input and one output

    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test the model
x_test = np.linspace(-10, 10, 100, dtype=np.float32)
x_test = torch.from_numpy(x_test.reshape(-1, 1))
y_test = model(x_test).detach().numpy()

# Plot the results
plt.scatter(x_train.numpy(), y_train.numpy(), label='Data')
plt.plot(x_test.numpy(), y_test, color='red', label='NN Approximation')
plt.legend()
plt.show()
