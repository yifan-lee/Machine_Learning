import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F  # activation function ReLU
from torch.optim import SGD  # stochastic gradient descent


from basic_NN import BasicNN_baseline, BasicNN, SimplifiedNN, BetterNN, flixNN, flixNN2





torch.manual_seed(0)
x_train = torch.linspace(0, 1, 1000)
noise_train = torch.randn(1000) * 0.1
y_train = (torch.sin(np.pi * x_train) > 0.5).float() + noise_train  # 二元分类目标

x_test = torch.linspace(0.05, 0.95, 20)
noise_test = torch.randn(20) * 0.1
y_test = (torch.sin(np.pi * x_test) > 0.5).float() + noise_test

### BasicNN_baseline

model = BasicNN_baseline()
output_values = model(x_test)

testLoss = F.mse_loss(output_values, y_test)
print(f"MSE for baseline NN model is {testLoss:.4f}")

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

### flixNN


model = flixNN(dim1=4,dim2=4)
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


### flixNN2


model = flixNN2(dim1=2)
criterion = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)


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
