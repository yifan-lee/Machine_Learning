import pandas as pd

import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.CNN import CNN, BetterCNN

from utils.train_model import train
from utils.eval_model import evaluate_flexible
from utils.load_data_from_csv import load_data_from_csv


def run_cnn(path,criterion, epochs,patience,device,predFunction):
    data = _load_data(path)
    
    model = CNN()
    _train_and_eval_model(model, data, criterion, epochs,patience,device,predFunction)
    
    model = BetterCNN()
    _train_and_eval_model(model, data, criterion, epochs,patience,device,predFunction)
    
def _load_data(path):
    transform = transforms.Compose([transforms.ToTensor()])
    trainDataset = datasets.EMNIST(
        root=path,
        split='letters',
        train=True,
        download=True,
        transform=transform
    )
    testDataset = datasets.EMNIST(
        root=path,
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
    data = {
        'trainCNNLoader': trainCNNLoader,
        'testCNNLoader': testCNNLoader
    }
    return data



def _train_and_eval_model(model, data, criterion, epochs,patience,device,predFunction):
    trainCNNLoader = data['trainCNNLoader']
    testCNNLoader = data['testCNNLoader']
    optimizer = Adam(model.parameters(), lr=1e-3)
    modelTrained = train(
        model=model, 
        x=trainCNNLoader, 
        y=None, 
        optimizer=optimizer, 
        criterion=criterion, 
        epochs=epochs, 
        patience=patience, 
        device=device
    )
    correctPercent, wrongIndexes = evaluate_flexible(
        model=modelTrained, 
        x=testCNNLoader, 
        y=None, 
        criterion=criterion, 
        device=device,
        predFunction=predFunction
    )
    print(f"Correct percent for {model.__class__.__name__} model is {correctPercent*100:.2f} %")
    return modelTrained