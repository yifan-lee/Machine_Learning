import numpy as np
import pandas as pd
import torch
# import torch.nn as nn
# import torch.nn.functional as F  # activation function ReLU
from torch.optim import SGD  # stochastic gradient descent

from utils.train_model import train
from utils.eval_model import evaluate
from model.basic_NN import BasicNN_baseline, BasicNN, SimplifiedNN, BetterNN, flixNN, flixNN2
from model.basic_NN_2dim_inputs_1dim_outputs import SimplifiedNN_dim2, SimplifiedNN_dim2_layer2
from model.basic_NN_2dim_inputs_3dim_1class_outputs import SimplifiedNN_dim2dim3, SimplifiedNN_dim2dim3_layer2




## Load data
xTrain = pd.read_csv('data/x_train_1d1d.csv', header=None).values
yTrain = pd.read_csv('data/y_train_1d1d.csv', header=None).values
xTest = pd.read_csv('data/x_test_1d1d.csv', header=None).values
yTest = pd.read_csv('data/y_test_1d1d.csv', header=None).values

xTrain = torch.tensor(xTrain, dtype=torch.float32)
yTrain = torch.tensor(yTrain, dtype=torch.float32)
xTest = torch.tensor(xTest, dtype=torch.float32)
yTest = torch.tensor(yTest, dtype=torch.float32)

xTrain2d = pd.read_csv('data/x_train_2d1d.csv', header=None).values
yTrain2d = pd.read_csv('data/y_train_2d1d.csv', header=None).values
xTest2d = pd.read_csv('data/x_test_2d1d.csv', header=None).values
yTest2d = pd.read_csv('data/y_test_2d1d.csv', header=None).values

xTrain2d = torch.tensor(xTrain2d, dtype=torch.float32)
yTrain2d = torch.tensor(yTrain2d, dtype=torch.float32)
xTest2d = torch.tensor(xTest2d, dtype=torch.float32)
yTest2d = torch.tensor(yTest2d, dtype=torch.float32)


## 1 dim

if 0:
    epochs = 1000
    criterion=torch.nn.MSELoss()

    model = BasicNN()
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
    loss = evaluate(model, xTest, yTest, criterion)
    print(f"MSE for baseline NN model is {loss:.4f}")

    model = SimplifiedNN()
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
    loss = evaluate(model, xTest, yTest, criterion)
    print(f"MSE for baseline SimplifiedNN model is {loss:.4f}")

    model = BetterNN()
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
    loss = evaluate(model, xTest, yTest, criterion)
    print(f"MSE for baseline BetterNN model is {loss:.4f}")

    model = flixNN(dim1=4,dim2=4)
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
    loss = evaluate(model, xTest, yTest, criterion)
    print(f"MSE for baseline flixNN model is {loss:.4f}")



## 2 dim

epochs = 1000
criterion=torch.nn.MSELoss()

model = SimplifiedNN_dim2(dim1=3)
optimizer=SGD(model.parameters(), lr=0.01)
modelTrained = train(model, xTrain2d, yTrain2d, optimizer, criterion, epochs)
loss = evaluate(model, xTest2d, yTest2d, criterion)
print(f"MSE for baseline SimplifiedNN_dim2 model is {loss:.4f}")

model = SimplifiedNN_dim2_layer2(dim1=2, dim2=5)
optimizer=SGD(model.parameters(), lr=0.01)
modelTrained = train(model, xTrain2d, yTrain2d, optimizer, criterion, epochs)
loss = evaluate(model, xTest2d, yTest2d, criterion)
print(f"MSE for baseline SimplifiedNN_dim2_layer2 model is {loss:.4f}")

# if model_class == 'dim2':
#     data = dat_2dim
#     n_epochs = 1000

#     ### SimplifiedNN_dim2

#     

#     ### SimplifiedNN_dim2_layer2

#     model = SimplifiedNN_dim2_layer2(dim1=2, dim2=5)
#     test_model(model, data, n_epochs)

# ## 2 dim 3 dim
# if model_class == 'dim2dim3':
#     data = dat_2dim3dim
#     n_epochs = 1000

#     ### SimplifiedNN_dim2dim3

#     model = SimplifiedNN_dim2dim3(dim1=3, num_classes=3)
#     test_model(model, data, n_epochs, criterion=nn.CrossEntropyLoss())

#     ### SimplifiedNN_dim2_layer2

#     model = SimplifiedNN_dim2_layer2(dim1=2, dim2=5, num_classes=3)
#     test_model(model, data, n_epochs, criterion=nn.CrossEntropyLoss())


# print("Success!")
