import pandas as pd
import numpy as np

import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.RNN import RNN_TS

from utils.train_model import train
from utils.eval_model import evaluate2, evaluate_flexible
from utils.word_to_index import tokenize
from utils.DataSetForDataLoader import numericDataset

from sklearn.model_selection import train_test_split



def run_rnn(path,featureName,seq_len,criterion, epochs,patience,device,predFunction,printLoss):
    data = _load_fincial_data(path,featureName,seq_len)
    
    xBatch, _ = next(iter(data['trainDataLoader']))
    featureDim = xBatch.shape[2]
    model = RNN_TS(featureDim=featureDim, hiddenDim=128, outputDim=featureDim)
    _train_and_eval_model(model, data, criterion, epochs,patience,device,predFunction,printLoss)
    



    
def _load_fincial_data(dataPath, featureName, seq_len=60):
    df = pd.read_csv(dataPath, header=[0,1], index_col=0)
    prices = df.loc[:, (slice(None), featureName)]
    returns = _get_return_from_fincial_data(prices)
    x, y = _create_sequences(returns, seq_len)
    data = _transfer_xy_to_DataLoader(x,y)
    return data
    
def _get_return_from_fincial_data(prices):
    prices = prices.ffill().dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    returns = returns.astype(np.float32)
    return returns

def _create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data.iloc[i:i+seq_len])
        y.append(data.iloc[i+seq_len])
    return np.array(X), np.array(y)

def _transfer_xy_to_DataLoader(x,y,xType=torch.float32, yType=torch.float32,test_size=0.2,batch_size=32):
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=test_size)
    trainData = numericDataset(xTrain, yTrain, xType, yType)
    testData = numericDataset(xTest, yTest, xType, yType)
    trainDataLoader = DataLoader(trainData, batch_size, shuffle=True)
    testDataLoader = DataLoader(testData, batch_size)
    data = {
        'trainDataLoader':trainDataLoader,
        'testDataLoader':testDataLoader
    }
    return data

def _train_and_eval_model(model, data, criterion, epochs,patience,device,predFunction,printLoss):
    trainCNNLoader = data['trainDataLoader']
    testCNNLoader = data['testDataLoader']
    optimizer = Adam(model.parameters(), lr=1e-3)
    modelTrained = train(
        model=model, 
        x=trainCNNLoader, 
        y=None, 
        optimizer=optimizer, 
        criterion=criterion, 
        epochs=epochs, 
        patience=patience, 
        device=device,
        printLoss = True,
    )
    loss = evaluate2(
        model=modelTrained, 
        x=testCNNLoader, 
        y=None, 
        criterion=criterion, 
        device=device,
        predFunction=predFunction
    )
    print(f"MSE for {model.__class__.__name__} model is {loss:.4f}")
    return modelTrained
