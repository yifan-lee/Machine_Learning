import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

import torch
from torch.optim import SGD  # stochastic gradient descent
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


from utils.train_model import train
from utils.eval_model import evaluate, evaluate_CNN, evaluate_RNN
from utils.draw_figure import prepare_data_for_draw_CNN_incorrect_predictions, draw_CNN_incorrect_predictions
from utils.word_to_index import tokenize
from utils.DataSetForDataLoader import TextDataset
from model.basic_NN import nn_baseline, nn_basic, nn_simple, nn_layer1, nn_layer1, nn_layer2
from model.basic_NN_2dim_inputs_1dim_outputs import NN_dim2, NN_dim2_layer2, NN_dim2_flixible_layer, NN_dim2_flixible_layer_dropout
from model.basic_NN_2dim_inputs_3dim_1class_outputs import nn_dim3c1,nn_dim3c1_dropout,nn_dim3c1_dropout_sequential
from model.CNN import CNN, BetterCNN
from model.RNN import BasicRNN

from sklearn.model_selection import train_test_split


## Settings
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

mod = 'nn_2d1d'

if mod == 'nn_1d1d':
    from run.run_nn_1d1d import run_nn_1d1d
    dataPath = {
            'xTrain': './data/NN/x_train_1d1d.csv',
            'yTrain': './data/NN/y_train_1d1d.csv',
            'xTest': './data/NN/x_test_1d1d.csv',
            'yTest': './data/NN/y_test_1d1d.csv'
        }
    epochs = 1000
    criterion=torch.nn.MSELoss()
    run_nn_1d1d(dataPath, epochs, criterion)

# if mod == 'nn_1d1d':
    
#     xTrain = pd.read_csv(path['xTrain'], header=None).values
#     yTrain = pd.read_csv(path['yTrain'], header=None).values
#     xTest = pd.read_csv(path['xTest'], header=None).values
#     yTest = pd.read_csv(path['yTest'], header=None).values

#     xTrain = torch.tensor(xTrain, dtype=torch.float32)
#     yTrain = torch.tensor(yTrain, dtype=torch.float32)
#     xTest = torch.tensor(xTest, dtype=torch.float32)
#     yTest = torch.tensor(yTest, dtype=torch.float32)

if mod == 'nn_2d1d':
    from run.run_nn_2d1d import run_nn_2d1d
    dataPath = {
            'xTrain': './data/NN/x_train_2d1d.csv',
            'yTrain': './data/NN/y_train_2d1d.csv',
            'xTest': './data/NN/x_test_2d1d.csv',
            'yTest': './data/NN/y_test_2d1d.csv'
        }
    epochs = 1000
    criterion=torch.nn.MSELoss()
    run_nn_2d1d(dataPath, epochs, criterion)
    
    
    


if mod == 'nn_ndnc':
    xTrainndnc = pd.read_csv('data/x_train_ndnc.csv', header=None).values
    yTrainndnc = pd.read_csv('data/y_train_ndnc.csv', header=None).values
    xTestndnc = pd.read_csv('data/x_test_ndnc.csv', header=None).values
    yTestndnc = pd.read_csv('data/y_test_ndnc.csv', header=None).values

    xTrainndnc = torch.tensor(xTrainndnc, dtype=torch.float32)
    yTrainndnc = torch.tensor(yTrainndnc, dtype=torch.int64).squeeze()
    xTestndnc = torch.tensor(xTestndnc, dtype=torch.float32)
    yTestndnc = torch.tensor(yTestndnc, dtype=torch.int64).squeeze()

if mod == 'CNN':
    transform = transforms.Compose([transforms.ToTensor()])
    trainDataset = datasets.EMNIST(
        root='./data',
        split='letters',
        train=True,
        download=True,
        transform=transform
    )
    testDataset = datasets.EMNIST(
        root='./data',
        split='letters',
        train=False,
        download=True,
        transform=transform
    )
    # Adjust labels: EMNIST 'letters' split labels go from 1 to 26, so subtract 1
    trainDataset.targets -= 1
    testDataset.targets -= 1
    batchSize = 32
    trainCNNLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testCNNLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)



if mod == 'RNN':

    df = pd.read_csv("./data/RNN/IMDB/imdb_train.csv")  # text,label
    texts = df["text"].values
    labels = df["label"].values

    max_vocab = 5000
    max_len = 100

    encodedTexts, word2idx = tokenize(texts, max_vocab, max_len)
    encodedTextsArray = np.array(encodedTexts, dtype=np.float32)
    encodedTexts3d = encodedTextsArray[:, :, None]
    xTrain, xTest, yTrain, yTest = train_test_split(encodedTexts3d, labels, test_size=0.2)
    trainData = TextDataset(xTrain, yTrain, torch.float32, torch.float32)
    testData = TextDataset(xTest, yTest, torch.float32, torch.float32)
    trainDataLoader = DataLoader(trainData, batch_size=32, shuffle=True)
    testDataLoader = DataLoader(testData, batch_size=32)


## 1 dim

# if mod == 'nn_1d1d':
#     epochs = 1000
#     criterion=torch.nn.MSELoss()

#     model = nn_basic()
#     optimizer=SGD(model.parameters(), lr=0.01)
#     modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
#     loss = evaluate(model, xTest, yTest, criterion)
#     print(f"MSE for nn_basic model is {loss:.4f}")

#     model = nn_simple()
#     optimizer=SGD(model.parameters(), lr=0.01)
#     modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
#     loss = evaluate(model, xTest, yTest, criterion)
#     print(f"MSE for nn_simple model is {loss:.4f}")

#     model = nn_layer1(hidden_dim=5)
#     optimizer=SGD(model.parameters(), lr=0.01)
#     modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
#     loss = evaluate(model, xTest, yTest, criterion)
#     print(f"MSE for nn_layer1 model is {loss:.4f}")

#     model = nn_layer2(dim1=4,dim2=4)
#     optimizer=SGD(model.parameters(), lr=0.01)
#     modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
#     loss = evaluate(model, xTest, yTest, criterion)
#     print(f"MSE for nn_layer2 model is {loss:.4f}")
    



## 2 dim

# if mod == 'nn_2d1d':
#     epochs = 5000
#     criterion=torch.nn.MSELoss()

#     model = NN_dim2(dim1=3)
#     optimizer=SGD(model.parameters(), lr=0.01)
#     modelTrained = train(model, xTrain2d, yTrain2d, optimizer, criterion, epochs)
#     loss = evaluate(model, xTest2d, yTest2d, criterion)
#     print(f"MSE for NN_dim2 model is {loss:.4f}")

#     model = NN_dim2_layer2(dim1=2, dim2=5)
#     optimizer=SGD(model.parameters(), lr=0.01)
#     modelTrained = train(model, xTrain2d, yTrain2d, optimizer, criterion, epochs)
#     loss = evaluate(model, xTest2d, yTest2d, criterion)
#     print(f"MSE for NN_dim2_layer2 model is {loss:.4f}")

#     model = NN_dim2_flixible_layer(dims=[5, 10, 3])
#     optimizer=SGD(model.parameters(), lr=0.01)
#     modelTrained = train(model, xTrain2d, yTrain2d, optimizer, criterion, epochs)
#     loss = evaluate(model, xTest2d, yTest2d, criterion)
#     print(f"MSE for NN_dim2_flixible_layer model is {loss:.4f}")

#     model = NN_dim2_flixible_layer_dropout(dims=[8, 10, 4], dropoutRate=0.5)
#     optimizer=SGD(model.parameters(), lr=0.01)
#     modelTrained = train(model, xTrain2d, yTrain2d, optimizer, criterion, epochs)
#     loss = evaluate(model, xTest2d, yTest2d, criterion)
#     print(f"MSE for NN_dim2_flixible_layer_dropout model is {loss:.4f}")

## 3 dim 10 classes

if mod == 'nn_ndnc':
    epochs = 500
    criterion=torch.nn.CrossEntropyLoss()

    model = nn_dim3c1(dims=[20,20,20])
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrainndnc, yTrainndnc, optimizer, criterion, epochs)
    loss = evaluate(model, xTestndnc, yTestndnc, criterion)
    print(f"Cross entropy for nn_dim3c1 model is {loss:.4f}")
    
    model = nn_dim3c1_dropout(dims=[32,16], dropoutRate=0.5)
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrainndnc, yTrainndnc, optimizer, criterion, epochs)
    loss = evaluate(model, xTestndnc, yTestndnc, criterion)
    print(f"Cross entropy for nn_dim3c1_dropout model is {loss:.4f}")
    
    model = nn_dim3c1_dropout_sequential(dims=[32,16], dropoutRate=0.5)
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrainndnc, yTrainndnc, optimizer, criterion, epochs)
    loss = evaluate(model, xTestndnc, yTestndnc, criterion)
    print(f"Cross entropy for nn_dim3c1_dropout_sequential model is {loss:.4f}")

    # for dims in [[32,16], [64,32], [64,64,32]]:
    #     for dr in [0.1, 0.3, 0.5]:
    #         model = nn_dim3c1_dropout(dims=dims, dropoutRate=dr)
    #         optimizer=SGD(model.parameters(), lr=0.01)
    #         modelTrained = train(model, xTrainndnc, yTrainndnc, optimizer, criterion, epochs)
    #         loss = evaluate(model, xTestndnc, yTestndnc, criterion)
    #         print(f"Cross entropy for dims: {dims} and dropout rate: {dr} model is {loss:.4f}")


## CNN
if mod == 'CNN':
    epochs = 10
    patience = 1
    
    criterion = nn.CrossEntropyLoss()
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model = train(
        model=model, 
        x=trainCNNLoader, 
        y=None, 
        optimizer=optimizer, 
        criterion=criterion, 
        epochs=epochs, 
        patience=patience, 
        device=device
    )
    correctPercent, wrongIndexes = evaluate_CNN(
        model=model, 
        x=testCNNLoader, 
        y=None, 
        criterion=criterion, 
        device=device
    )
    print(f"Cross entropy for CNN model is {correctPercent*100:.2f} %")
    dataForFigure = prepare_data_for_draw_CNN_incorrect_predictions(testDataset, wrongIndexes)
    draw_CNN_incorrect_predictions(dataForFigure, figurePath='./figures', fileName='CNN_wrong_predictions')


    model = BetterCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model = train(
        model=model, 
        x=trainCNNLoader, 
        y=None, 
        optimizer=optimizer, 
        criterion=criterion, 
        epochs=epochs, 
        patience=patience, 
        device=device
    )
    correctPercent, wrongIndexes = evaluate_CNN(
        model=model, 
        x=testCNNLoader, 
        y=None, 
        criterion=criterion, 
        device=device
    )
    print(f"Cross entropy for CNN model is {correctPercent*100:.2f} %")
    dataForFigure = prepare_data_for_draw_CNN_incorrect_predictions(testDataset, wrongIndexes)
    draw_CNN_incorrect_predictions(dataForFigure, figurePath='./figures', fileName='BetterCNN_wrong_predictions')
    
    
    
    
    
## RNN
if mod == 'RNN':
    epochs = 5
    model = BasicRNN(featureDim=1, hiddenDim=128, outputDim=1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    modelTrained = train(
        model = model, 
        x = trainDataLoader, 
        optimizer = optimizer, 
        criterion = criterion, 
        epochs = epochs,
        printLoss = True,
    )
    
    correctPercent, wrongIndexes = evaluate_RNN(
        model = modelTrained, 
        x = testDataLoader, 
        device = device
    )

    print(f"Correct prediction percentage: {correctPercent * 100:.2f}%")
print("Success!")
