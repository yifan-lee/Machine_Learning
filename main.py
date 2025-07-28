import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F  # activation function ReLU
from torch.optim import SGD  # stochastic gradient descent


from basic_NN import BasicNN_baseline, BasicNN, SimplifiedNN, BetterNN, flixNN, flixNN2
from basic_NN_2dim_inputs import SimplifiedNN_dim2, SimplifiedNN_dim2_layer2
from basic_NN_2dim_inputs_3dim_outputs import SimplifiedNN_dim2dim3, SimplifiedNN_dim2dim3_layer2

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

dat_1dim = {
    'x_train':x_train,
    'y_train':y_train,
    'x_test':x_test,
    'y_test':y_test
}

x_train = torch.rand(1000, 2)
y_train = (((torch.sin(np.pi * x_train[:, 0]) + torch.cos(np.pi * x_train[:, 1])) > 1.0).float() + torch.randn(1000) * 0.1)

x_test = torch.rand(200, 2)
y_test = (((torch.sin(np.pi * x_test[:, 0]) + torch.cos(np.pi * x_test[:, 1])) > 1.0).float() + torch.randn(200) * 0.1)

dat_2dim = {
    'x_train':x_train,
    'y_train':y_train,
    'x_test':x_test,
    'y_test':y_test
}


def label(x):
    if x[0]+ x[1] > 1:
        return 0
    elif x[0]/3 + x[1]/2 > 0.5:
        return 1
    else:
        return 2
    
x_train = torch.rand(1000, 2)
y_train = torch.tensor([label(x) for x in x_train])
x_test = torch.rand(200, 2)
y_test = torch.tensor([label(x) for x in x_test])
dat_2dim3dim = {
    'x_train':x_train,
    'y_train':y_train,
    'x_test':x_test,
    'y_test':y_test
}

def test_model(model, data, n_epochs, criterion=nn.MSELoss()):
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
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
    print(f"MSE (Train) for baseline NN model is {loss.item():.4f}")
    print(f"MSE (Test) for baseline NN model is {lossTest.item():.4f}")
    
    return 0

model_class = 'dim2dim3'

## 1 dim
if model_class == 'dim1':
    
    ### BasicNN_baseline

    model = BasicNN_baseline()
    output_values = model(x_test)

    testLoss = F.mse_loss(output_values, y_test)
    print(f"MSE for baseline NN model is {testLoss:.4f}")



    n_epochs = 1000
    data = dat_1dim

    ### BasicNN

    model = BasicNN()
    test_model(model, data, n_epochs)


    ### SimplifiedNN

    model = SimplifiedNN()
    test_model(model, data, n_epochs)


    ### BetterNN

    model = BetterNN()
    test_model(model, data, n_epochs)

    ### flixNN

    model = flixNN(dim1=4,dim2=4)
    test_model(model, data, n_epochs)

    ### flixNN2


    model = flixNN2(dim1=5)
    test_model(model, data, n_epochs)




## 2 dim

if model_class == 'dim2':
    data = dat_2dim
    n_epochs = 1000

    ### SimplifiedNN_dim2

    model = SimplifiedNN_dim2(dim1=3)
    test_model(model, data, n_epochs)

    ### SimplifiedNN_dim2_layer2

    model = SimplifiedNN_dim2_layer2(dim1=2, dim2=5)
    test_model(model, data, n_epochs)

## 2 dim 3 dim
if model_class == 'dim2dim3':
    data = dat_2dim3dim
    n_epochs = 1000

    ### SimplifiedNN_dim2dim3

    model = SimplifiedNN_dim2dim3(dim1=3, num_classes=3)
    test_model(model, data, n_epochs, criterion=nn.CrossEntropyLoss())

    ### SimplifiedNN_dim2_layer2

    model = SimplifiedNN_dim2_layer2(dim1=2, dim2=5, num_classes=3)
    test_model(model, data, n_epochs, criterion=nn.CrossEntropyLoss())


print("Success!")
