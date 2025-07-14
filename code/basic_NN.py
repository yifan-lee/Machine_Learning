import torch
import torch.nn as nn
import torch.nn.functional as F  # activation function ReLU


class BasicNN_baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(
            torch.tensor(1.7), requires_grad=False
        )  # don't optims it
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        self.final_bias = nn.Parameter(torch.tensor(-16.0), requires_grad=False)

    def forward(self, input):
        hiddenLayerNode1_0 = input * self.w00 + self.b00
        hiddenLayerNode1_1 = F.relu(hiddenLayerNode1_0)

        hiddenLayerNode2_0 = input * self.w10 + self.b10
        hiddenLayerNode2_1 = F.relu(hiddenLayerNode2_0)

        output = F.relu(
            self.w01 * hiddenLayerNode1_1
            + self.w11 * hiddenLayerNode2_1
            + self.final_bias
        )

        return output


class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.randn(1), requires_grad=True)  # don't optims it
        self.b00 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.w01 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.w10 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.b10 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.w11 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.final_bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, input):
        hiddenLayerNode1_0 = input * self.w00 + self.b00
        hiddenLayerNode1_1 = F.relu(hiddenLayerNode1_0)

        hiddenLayerNode2_0 = input * self.w10 + self.b10
        hiddenLayerNode2_1 = F.relu(hiddenLayerNode2_0)

        output = F.relu(
            self.w01 * hiddenLayerNode1_1
            + self.w11 * hiddenLayerNode2_1
            + self.final_bias
        )

        return output


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



class BetterNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.randn(1, 16))
        self.B1 = nn.Parameter(torch.randn(16))
        self.W2 = nn.Parameter(torch.randn(16, 16))
        self.B2 = nn.Parameter(torch.randn(16))
        self.W3 = nn.Parameter(torch.randn(16, 1))
        self.B3 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input.view(-1, 1)
        y1 = F.relu(x @ self.W1 + self.B1)
        y2 = F.relu(y1 @ self.W2 + self.B2)
        output = F.relu(y2 @ self.W3 + self.B3)
        # output = (y2 @ self.W3 + self.B3)
        return output

class flixNN(nn.Module):
    def __init__(self, dim1=16,dim2=16,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.randn(1, dim1))
        self.B1 = nn.Parameter(torch.randn(dim1))
        self.W2 = nn.Parameter(torch.randn(dim1, dim2))
        self.B2 = nn.Parameter(torch.randn(dim2))
        self.W3 = nn.Parameter(torch.randn(dim2, 1))
        self.B3 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input.view(-1, 1)
        y1 = F.relu(x @ self.W1 + self.B1)
        y2 = F.relu(y1 @ self.W2 + self.B2)
        output = F.relu(y2 @ self.W3 + self.B3)
        # output = (y2 @ self.W3 + self.B3)
        return output
    
class flixNN2(nn.Module):
    def __init__(self, dim1=2,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.randn(1, dim1))
        self.B1 = nn.Parameter(torch.randn(dim1))
        self.W3 = nn.Parameter(torch.randn(dim1, 1))
        self.B3 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input.view(-1, 1)
        y1 = F.relu(x @ self.W1 + self.B1)
        output = F.relu(y1 @ self.W3 + self.B3)
        return output
    
    
# import numpy as np
# import pandas as pd


# import torch
# import torch.nn as nn
# import torch.nn.functional as F  # activation function ReLU
# from torch.optim import SGD  # stochastic gradient descent


# torch.manual_seed(0)
# x_train = torch.linspace(0, 1, 1000)
# noise_train = torch.randn(1000) * 0.1
# y_train = (torch.sin(np.pi * x_train) > 0.5).float() + noise_train  # 二元分类目标

# x_test = torch.linspace(0.05, 0.95, 20)
# noise_test = torch.randn(20) * 0.1
# y_test = (torch.sin(np.pi * x_test) > 0.5).float() + noise_test


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

# model = flixNN2(dim1=2)
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