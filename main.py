from qiskit import QuantumCircuit, execute, Aer
import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    ZFeatureMap,
    RealAmplitudes,
)
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from matplotlib import pyplot as plt
from tqdm import tqdm
from math import sin

num_qubits = 1

def create_qnn(num_qubits):
    feature_map = ZFeatureMap(num_qubits, reps=1)
    ansatz = RealAmplitudes(num_qubits, reps=1)

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )

    return qnn


def defined_function(x):
    # Given a value x, return sin(x)
    return sin(x)
    
class build_qnn(nn.Module):
    def __init__(self, qnn):
        super(build_qnn, self).__init__()
        self.qnn = TorchConnector(qnn)
        #self.fc = nn.Linear(2, 1)
        #self.fc1 = nn.Linear(1, 1)
    
    def forward(self, x):
        x = self.qnn(x)
        #x = self.fc(x)
        #x = self.fc1(x)
        
        return x

def training_step(x_train, y_train, model, criterion, optimizer):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Input values for the defined function
x_train = np.arange(0, 50, 0.1, dtype=np.float32)
x_train = np.array(x_train)
y_train = np.zeros(len(x_train))

x_test = np.arange(0, 50, 0.1, dtype=np.float32)
x_test = np.array(x_test)
y_test = np.zeros(len(x_test))

for i in range(len(x_train)):
    y_train[i] = defined_function(x_train[i])
    
for i in range(len(x_test)):
    y_test[i] = defined_function(x_test[i])

plt.scatter(x_train, y_train)
plt.show()
x_train = torch.from_numpy(x_train.reshape(-1, 1))
y_train = torch.from_numpy(y_train.reshape(-1, 1))

for i in tqdm(range(20)):
    # Define the neural network model
    qnn = create_qnn(num_qubits)
    model = build_qnn(qnn)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Training loop
    for epoch in tqdm(range(10)):
        loss = training_step(x_train, y_train, model, criterion, optimizer)
    
# Test the model
x_test = torch.from_numpy(x_test.reshape(-1, 1))
y_test = model(x_test).detach().numpy()
    
# Plot the results with the ideal linear function
plt.scatter(x_train.numpy(), y_train.numpy(), label="Data")
plt.plot(x_test, y_test, color="red", label="NN Approximation")
plt.legend()
plt.title("Output of QNN for different input values")
plt.xlabel("Input value")
plt.ylabel("Output value")
plt.show()
