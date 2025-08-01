import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from utils.train_model import train
from utils.draw_figure import prepare_data_for_draw_CNN_incorrect_predictions, draw_CNN_incorrect_predictions
from utils.word_to_index import tokenize
from utils.DataSetForDataLoader import numericDataset


from sklearn.model_selection import train_test_split


## Settings
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

mod = 'RNN'

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
    run_nn_1d1d(dataPath, criterion, epochs)


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
    run_nn_2d1d(dataPath, criterion, epochs)
    

if mod == 'nn_ndnc':
    from run.run_nn_ndnc import run_nn_ndnc
    dataPath = {
            'xTrain': './data/NN/x_train_ndnc.csv',
            'yTrain': './data/NN/y_train_ndnc.csv',
            'xTest': './data/NN/x_test_ndnc.csv',
            'yTest': './data/NN/y_test_ndnc.csv'
        }
    epochs = 500
    criterion=torch.nn.CrossEntropyLoss()
    run_nn_ndnc(dataPath, criterion, epochs)
    
    
if mod == 'CNN':
    from run.run_cnn import run_cnn
    epochs = 10
    patience = 1
    criterion = nn.CrossEntropyLoss()
    dataPath = './data/CNN'
    def predict_CNN_result(output):
        return output.argmax(dim=1)
    predFunction =predict_CNN_result
    
    run_cnn(dataPath,criterion, epochs,patience,device,predFunction)
    

if mod == 'RNN':
    from run.run_rnn import run_rnn
    
    dataPath = r'./data/RNN/financial_timeseries.csv'
    featureName = ['Close', 'Open']
    seq_len = 64
    criterion = nn.MSELoss()
    epochs=5
    patience = 1
    def predict_RNN_TS_result(output):
        return output
    predFunction = predict_RNN_TS_result
    run_rnn(
        path = dataPath,
        featureName = featureName,
        seq_len = seq_len,
        criterion = criterion, 
        epochs=epochs,
        patience = patience,
        device=device,
        predFunction=predFunction,
        printLoss = True
    )
    
    # dataPath = "./data/RNN/IMDB/imdb_train.csv"
    # epochs = 10
    # patience = 1
    # criterion = nn.BCEWithLogitsLoss()
    # def predict_RNN_result(output):
    #     return (torch.sigmoid(output) > 0.5).float()
    # predFunction = predict_RNN_result
    # run_rnn(
    #     path = dataPath,
    #     criterion = criterion, 
    #     epochs=epochs,
    #     patience = patience,
    #     device=device,
    #     predFunction=predFunction,
    #     printLoss = True
    # )

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

# if mod == 'nn_ndnc':
#     epochs = 500
#     criterion=torch.nn.CrossEntropyLoss()

#     model = nn_dim3c1(dims=[20,20,20])
#     optimizer=SGD(model.parameters(), lr=0.01)
#     modelTrained = train(model, xTrainndnc, yTrainndnc, optimizer, criterion, epochs)
#     loss = evaluate(model, xTestndnc, yTestndnc, criterion)
#     print(f"Cross entropy for nn_dim3c1 model is {loss:.4f}")
    
#     model = nn_dim3c1_dropout(dims=[32,16], dropoutRate=0.5)
#     optimizer=SGD(model.parameters(), lr=0.01)
#     modelTrained = train(model, xTrainndnc, yTrainndnc, optimizer, criterion, epochs)
#     loss = evaluate(model, xTestndnc, yTestndnc, criterion)
#     print(f"Cross entropy for nn_dim3c1_dropout model is {loss:.4f}")
    
#     model = nn_dim3c1_dropout_sequential(dims=[32,16], dropoutRate=0.5)
#     optimizer=SGD(model.parameters(), lr=0.01)
#     modelTrained = train(model, xTrainndnc, yTrainndnc, optimizer, criterion, epochs)
#     loss = evaluate(model, xTestndnc, yTestndnc, criterion)
#     print(f"Cross entropy for nn_dim3c1_dropout_sequential model is {loss:.4f}")

    # for dims in [[32,16], [64,32], [64,64,32]]:
    #     for dr in [0.1, 0.3, 0.5]:
    #         model = nn_dim3c1_dropout(dims=dims, dropoutRate=dr)
    #         optimizer=SGD(model.parameters(), lr=0.01)
    #         modelTrained = train(model, xTrainndnc, yTrainndnc, optimizer, criterion, epochs)
    #         loss = evaluate(model, xTestndnc, yTestndnc, criterion)
    #         print(f"Cross entropy for dims: {dims} and dropout rate: {dr} model is {loss:.4f}")


## CNN
# if mod == 'CNN':
#     epochs = 10
#     patience = 1
    
#     criterion = nn.CrossEntropyLoss()
#     model = CNN()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     model = train(
#         model=model, 
#         x=trainCNNLoader, 
#         y=None, 
#         optimizer=optimizer, 
#         criterion=criterion, 
#         epochs=epochs, 
#         patience=patience, 
#         device=device
#     )
#     correctPercent, wrongIndexes = evaluate_CNN(
#         model=model, 
#         x=testCNNLoader, 
#         y=None, 
#         criterion=criterion, 
#         device=device
#     )
#     print(f"Cross entropy for CNN model is {correctPercent*100:.2f} %")
#     dataForFigure = prepare_data_for_draw_CNN_incorrect_predictions(testDataset, wrongIndexes)
#     draw_CNN_incorrect_predictions(dataForFigure, figurePath='./figures', fileName='CNN_wrong_predictions')


#     model = BetterCNN()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     model = train(
#         model=model, 
#         x=trainCNNLoader, 
#         y=None, 
#         optimizer=optimizer, 
#         criterion=criterion, 
#         epochs=epochs, 
#         patience=patience, 
#         device=device
#     )
#     correctPercent, wrongIndexes = evaluate_CNN(
#         model=model, 
#         x=testCNNLoader, 
#         y=None, 
#         criterion=criterion, 
#         device=device
#     )
#     print(f"Cross entropy for CNN model is {correctPercent*100:.2f} %")
#     dataForFigure = prepare_data_for_draw_CNN_incorrect_predictions(testDataset, wrongIndexes)
#     draw_CNN_incorrect_predictions(dataForFigure, figurePath='./figures', fileName='BetterCNN_wrong_predictions')
    
    
    
    
    
## RNN
# if mod == 'RNN':
#     epochs = 5
#     model = BasicRNN(featureDim=1, hiddenDim=128, outputDim=1)
#     model = model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
#     modelTrained = train(
#         model = model, 
#         x = trainDataLoader, 
#         optimizer = optimizer, 
#         criterion = criterion, 
#         epochs = epochs,
#         printLoss = True,
#     )
    
#     correctPercent, wrongIndexes = evaluate_RNN(
#         model = modelTrained, 
#         x = testDataLoader, 
#         device = device
#     )

#     print(f"Correct prediction percentage: {correctPercent * 100:.2f}%")
print("Success!")
