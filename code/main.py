import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F  # activation function ReLU
from torch.optim import SGD  # stochastic gradient descent


from basic_NN import BasicNN_baseline, BasicNN, SimplifiedNN, BetterNN


x_train = torch.linspace(0, 1, steps=100)
y_train = torch.sin(torch.pi * x_train)


x_test = torch.linspace(0.05, 0.95, steps=10)
torch.manual_seed(42)  # 保证可复现
noise = torch.randn(10) * 0.1
y_test = torch.sin(torch.pi * x_test) + noise


### BasicNN_baseline

model = BasicNN_baseline()
output_values = model(x_test)

testLoss = F.mse_loss(output_values, y_test)
print(f"MSE for baseline NN model is {testLoss}")

### BasicNN

model = BasicNN()
criterion = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)


n_epochs = 1000
loss_history = []

for epoch in range(n_epochs):
    # forward pass
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss_history.append(loss.item())

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if epoch % 100 == 0:
    #     print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

y_test_pred = model(x_test)
lossTest = criterion(y_test_pred, y_test)
print(f"MSE (Train) for baseline NN model is {loss.item():.4f}")
print(f"MSE (Test) for baseline NN model is {lossTest.item():.4f}")

# print("== BasicNN 参数 ==")
# for name, param in model.named_parameters():
#     print(f"{name}: {param.data}")

### SimplifiedNN


model = SimplifiedNN()
criterion = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)


n_epochs = 1000
loss_history = []

for epoch in range(n_epochs):
    # forward pass
    y_pred = model(x_train).view(-1)
    loss = criterion(y_pred, y_train)
    loss_history.append(loss.item())

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y_test_pred = model(x_test).view(-1)
lossTest = criterion(y_test_pred, y_test)
print(f"MSE (Train) for Simplified NN model is {loss.item():.4f}")
print(f"MSE (Test) for Simplified NN model is {lossTest.item():.4f}")

### BetterNN


model = BetterNN()
criterion = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)


n_epochs = 1000
loss_history = []

for epoch in range(n_epochs):
    # forward pass
    y_pred = model(x_train).view(-1)
    loss = criterion(y_pred, y_train)
    loss_history.append(loss.item())

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y_test_pred = model(x_test).view(-1)
lossTest = criterion(y_test_pred, y_test)
print(f"MSE (Train) for Better NN model is {loss.item():.4f}")
print(f"MSE (Test) for Better NN model is {lossTest.item():.4f}")

print("Success!")
