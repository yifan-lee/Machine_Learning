import pandas as pd

import torch
from torch.optim import SGD

from model.basic_NN import nn_baseline, nn_basic, nn_simple, nn_layer1, nn_layer1, nn_layer2
from utils.train_model import train
from utils.eval_model import evaluate
from utils.load_data_from_csv import load_data_from_csv

def run_nn_1d1d(path, criterion, epochs):
    dataRaw = load_data_from_csv(path)
    data = transfor_data_to_tensor(dataRaw)
    
    model = nn_basic()
    train_and_eval_model(model, data, criterion, epochs)
    
    model = nn_simple()
    train_and_eval_model(model, data, criterion, epochs)
    
    model = nn_layer1()
    train_and_eval_model(model, data, criterion, epochs)
    
    model = nn_layer2()
    train_and_eval_model(model, data, criterion, epochs)

    
    


def transfor_data_to_tensor(dataRaw):
    xTrain = torch.tensor(dataRaw['xTrain'], dtype=torch.float32)
    yTrain = torch.tensor(dataRaw['yTrain'], dtype=torch.float32)
    xTest = torch.tensor(dataRaw['xTest'], dtype=torch.float32)
    yTest = torch.tensor(dataRaw['yTest'], dtype=torch.float32)
    data = {
        'xTrain': xTrain,
        'yTrain': yTrain,
        'xTest': xTest,
        'yTest': yTest
    }
    return data

def train_and_eval_model(model, data, criterion, epochs):
    xTrain = data['xTrain']
    yTrain = data['yTrain']
    xTest = data['xTest']
    yTest = data['yTest']
    optimizer=SGD(model.parameters(), lr=0.01)
    modelTrained = train(model, xTrain, yTrain, optimizer, criterion, epochs)
    loss = evaluate(modelTrained, xTest, yTest, criterion)
    print(f"MSE for {model.__class__.__name__} model is {loss:.4f}")
    return modelTrained

