import pandas as pd

import torch
from torch.optim import SGD

from model.basic_NN_2dim_inputs_3dim_1class_outputs import nn_dim3c1,nn_dim3c1_dropout,nn_dim3c1_dropout_sequential

from utils.train_model import train
from utils.eval_model import evaluate
from utils.load_data_from_csv import load_data_from_csv


def run_nn_ndnc(path, criterion, epochs):
    dataRaw = load_data_from_csv(path)
    data = transfor_data_to_tensor(dataRaw)
    
    model = nn_dim3c1(dims=[20,20,20])
    train_and_eval_model(model, data, criterion, epochs)
    
    model = nn_dim3c1_dropout(dims=[32,16], dropoutRate=0.5)
    train_and_eval_model(model, data, criterion, epochs)
    
    model = nn_dim3c1_dropout_sequential(dims=[32,16], dropoutRate=0.5)
    train_and_eval_model(model, data, criterion, epochs)
    
    

def transfor_data_to_tensor(dataRaw):
    xTrain = torch.tensor(dataRaw['xTrain'], dtype=torch.float32)
    yTrain = torch.tensor(dataRaw['yTrain'], dtype=torch.int64)
    xTest = torch.tensor(dataRaw['xTest'], dtype=torch.float32)
    yTest = torch.tensor(dataRaw['yTest'], dtype=torch.int64)
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
    print(f"Cross entropy for {model.__class__.__name__} model is {loss:.4f}")
    return modelTrained