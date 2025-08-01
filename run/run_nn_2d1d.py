import pandas as pd

import torch
from torch.optim import SGD

from model.basic_NN_2dim_inputs_1dim_outputs import NN_dim2, NN_dim2_layer2, NN_dim2_flixible_layer, NN_dim2_flixible_layer_dropout

from utils.train_model import train
from utils.eval_model import evaluate
from utils.load_data_from_csv import load_data_from_csv

def run_nn_2d1d(path, criterion, epochs):
    dataRaw = load_data_from_csv(path)
    data = transfor_data_to_tensor(dataRaw)
    
    model = NN_dim2(dim1=3)
    train_and_eval_model(model, data, criterion, epochs)
    
    model = NN_dim2_layer2(dim1=2, dim2=5)
    train_and_eval_model(model, data, criterion, epochs)
    
    model = NN_dim2_flixible_layer(dims=[5, 10, 3])
    train_and_eval_model(model, data, criterion, epochs)
    
    model = NN_dim2_flixible_layer_dropout(dims=[8, 10, 4], dropoutRate=0.5)
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
