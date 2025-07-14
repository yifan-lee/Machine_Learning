import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F  # activation function ReLU
from torch.optim import SGD  # stochastic gradient descent


from basic_NN import BasicNN_baseline, BasicNN, SimplifiedNN, BetterNN, flixNN, flixNN2
from basic_NN_2dim_inputs import SimplifiedNN_dim2

class SimplifiedNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.randn(1, 2))
        self.B1 = nn.Parameter(torch.randn(2))
        self.W2 = nn.Parameter(torch.randn(2, 1))
        self.B2 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input.view(-1, 1)
        y1 = F.relu(x @ self.W1 + self.B1)
        output = F.relu(y1 @ self.W2 + self.B2)
        return output



torch.manual_seed(0)
x_train = torch.linspace(0, 1, 1000)
noise_train = torch.randn(1000) * 0.1
y_train = (torch.sin(np.pi * x_train) > 0.5).float() + noise_train  # 二元分类目标

x_test = torch.linspace(0.05, 0.95, 20)
noise_test = torch.randn(20) * 0.1
y_test = (torch.sin(np.pi * x_test) > 0.5).float() + noise_test


x2_train = torch.rand(1000, 2)
y2_train = (((torch.sin(np.pi * x2_train[:, 0]) + torch.cos(np.pi * x2_train[:, 1])) > 1.0).float() + torch.randn(1000) * 0.1)

x2_test = torch.rand(200, 2)
y2_test = (((torch.sin(np.pi * x2_test[:, 0]) + torch.cos(np.pi * x2_test[:, 1])) > 1.0).float() + torch.randn(200) * 0.1)

# ### BasicNN_baseline

# model = BasicNN_baseline()
# output_values = model(x_test)

# testLoss = F.mse_loss(output_values, y_test)
# print(f"MSE for baseline NN model is {testLoss:.4f}")

# ### BasicNN

# model = BasicNN()
# criterion = nn.MSELoss()
# optimizer = SGD(model.parameters(), lr=0.1)

# n_epochs = 1000
# loss_history = []

# for epoch in range(n_epochs):
#     # forward pass
#     y_pred = model(x_train)
#     loss = criterion(y_pred, y_train)
#     loss_history.append(loss.item())

#     # backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# y_test_pred = model(x_test)
# lossTest = criterion(y_test_pred, y_test)
# print(f"MSE (Train) for baseline NN model is {loss.item():.4f}")
# print(f"MSE (Test) for baseline NN model is {lossTest.item():.4f}")

# # print("== BasicNN 参数 ==")
# # for name, param in model.named_parameters():
# #     print(f"{name}: {param.data}")

# ### SimplifiedNN


# model = SimplifiedNN()
# criterion = nn.MSELoss()
# optimizer = SGD(model.parameters(), lr=0.1)


# n_epochs = 1000
# loss_history = []

# for epoch in range(n_epochs):
#     # forward pass
#     y_pred = model(x_train).view(-1)
#     loss = criterion(y_pred, y_train)
#     loss_history.append(loss.item())

#     # backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# y_test_pred = model(x_test).view(-1)
# lossTest = criterion(y_test_pred, y_test)
# print(f"MSE (Train) for Simplified NN model is {loss.item():.4f}")
# print(f"MSE (Test) for Simplified NN model is {lossTest.item():.4f}")

# ### BetterNN


# model = BetterNN()
# criterion = nn.MSELoss()
# optimizer = SGD(model.parameters(), lr=0.1)


# n_epochs = 1000
# loss_history = []

# for epoch in range(n_epochs):
#     # forward pass
#     y_pred = model(x_train).view(-1)
#     loss = criterion(y_pred, y_train)
#     loss_history.append(loss.item())

#     # backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# y_test_pred = model(x_test).view(-1)
# lossTest = criterion(y_test_pred, y_test)
# print(f"MSE (Train) for Better NN model is {loss.item():.4f}")
# print(f"MSE (Test) for Better NN model is {lossTest.item():.4f}")

# ### flixNN


# model = flixNN(dim1=4,dim2=4)
# criterion = nn.MSELoss()
# optimizer = SGD(model.parameters(), lr=0.1)


# n_epochs = 1000
# loss_history = []

# for epoch in range(n_epochs):
#     # forward pass
#     y_pred = model(x_train).view(-1)
#     loss = criterion(y_pred, y_train)
#     loss_history.append(loss.item())

#     # backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# y_test_pred = model(x_test).view(-1)
# lossTest = criterion(y_test_pred, y_test)
# print(f"MSE (Train) for Better NN model is {loss.item():.4f}")
# print(f"MSE (Test) for Better NN model is {lossTest.item():.4f}")


### flixNN2


# model = flixNN2(dim1=5)
# criterion = nn.MSELoss()
# optimizer = SGD(model.parameters(), lr=0.01)


# n_epochs = 1000
# loss_history = []

# for epoch in range(n_epochs):
#     # forward pass
#     y_pred = model(x_train).view(-1)
#     loss = criterion(y_pred, y_train)
#     loss_history.append(loss.item())

#     # backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# y_test_pred = model(x_test).view(-1)
# lossTest = criterion(y_test_pred, y_test)
# print(f"MSE (Train) for Better NN model is {loss.item():.4f}")
# print(f"MSE (Test) for Better NN model is {lossTest.item():.4f}")


### SimplifiedNN_dim2


model = SimplifiedNN_dim2(dim1=3)
criterion = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)


n_epochs = 1000
loss_history = []

for epoch in range(n_epochs):
    # forward pass
    y2_pred = model(x2_train).view(-1)
    loss = criterion(y2_pred, y2_train)
    loss_history.append(loss.item())

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y2_test_pred = model(x2_test).view(-1)
lossTest = criterion(y2_test_pred, y2_test)
print(f"MSE (Train) for SimplifiedNN_dim2 model is {loss.item():.4f}")
print(f"MSE (Test) for SimplifiedNN_dim2 model is {lossTest.item():.4f}")

print("Success!")
