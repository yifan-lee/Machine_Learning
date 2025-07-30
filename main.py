import numpy as np
import pandas as pd
import torch
# import torch.nn as nn
# import torch.nn.functional as F  # activation function ReLU
from torch.optim import SGD  # stochastic gradient descent

from utils.train_model import train
from utils.eval_model import evaluate
from model.basic_NN import nn_baseline, nn_basic, nn_simple, nn_layer1, nn_layer1, nn_layer2
from model.basic_NN_2dim_inputs_1dim_outputs import NN_dim2, NN_dim2_layer2, NN_dim2_flixible_layer, NN_dim2_flixible_layer_dropout
from model.basic_NN_2dim_inputs_3dim_1class_outputs import nn_dim3c1




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


xTrainndnc = pd.read_csv('data/x_train_ndnc.csv', header=None).values
yTrainndnc = pd.read_csv('data/y_train_ndnc.csv', header=None).values
xTestndnc = pd.read_csv('data/x_test_ndnc.csv', header=None).values
yTestndnc = pd.read_csv('data/y_test_ndnc.csv', header=None).values

xTrainndnc = torch.tensor(xTrainndnc, dtype=torch.float32)
yTrainndnc = torch.tensor(yTrainndnc, dtype=torch.int64).squeeze()
xTestndnc = torch.tensor(xTestndnc, dtype=torch.float32)
yTestndnc = torch.tensor(yTestndnc, dtype=torch.int64).squeeze()



## 1 dim

if 0:
    epochs = 1000
    criterion=torch.nn.MSELoss()

    model = nn_basic()
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
    loss = evaluate(model, xTest, yTest, criterion)
    print(f"MSE for baseline nn_basic model is {loss:.4f}")

    model = nn_simple()
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
    loss = evaluate(model, xTest, yTest, criterion)
    print(f"MSE for baseline nn_simple model is {loss:.4f}")

    model = nn_layer1(hidden_dim=5)
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
    loss = evaluate(model, xTest, yTest, criterion)
    print(f"MSE for baseline nn_layer1 model is {loss:.4f}")

    model = nn_layer2(dim1=4,dim2=4)
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
    loss = evaluate(model, xTest, yTest, criterion)
    print(f"MSE for baseline nn_layer2 model is {loss:.4f}")
    



## 2 dim

if 0:
    epochs = 5000
    criterion=torch.nn.MSELoss()

    model = NN_dim2(dim1=3)
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain2d, yTrain2d, optimizer, criterion, epochs)
    loss = evaluate(model, xTest2d, yTest2d, criterion)
    print(f"MSE for baseline NN_dim2 model is {loss:.4f}")

    model = NN_dim2_layer2(dim1=2, dim2=5)
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain2d, yTrain2d, optimizer, criterion, epochs)
    loss = evaluate(model, xTest2d, yTest2d, criterion)
    print(f"MSE for baseline NN_dim2_layer2 model is {loss:.4f}")

    model = NN_dim2_flixible_layer(dims=[5, 10, 3])
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain2d, yTrain2d, optimizer, criterion, epochs)
    loss = evaluate(model, xTest2d, yTest2d, criterion)
    print(f"MSE for baseline NN_dim2_flixible_layer model is {loss:.4f}")

    model = NN_dim2_flixible_layer_dropout(dims=[8, 10, 4], dropoutRate=0.5)
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain2d, yTrain2d, optimizer, criterion, epochs)
    loss = evaluate(model, xTest2d, yTest2d, criterion)
    print(f"MSE for baseline NN_dim2_flixible_layer_dropout model is {loss:.4f}")

if 1:
    epochs = 1000
    criterion=torch.nn.CrossEntropyLoss()

    model = nn_dim3c1(dims=[2,2])
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrainndnc, yTrainndnc, optimizer, criterion, epochs)
    loss = evaluate(model, xTestndnc, yTestndnc, criterion)
    print(f"MSE for baseline nn_dim3c1 model is {loss:.4f}")

print("Success!")
