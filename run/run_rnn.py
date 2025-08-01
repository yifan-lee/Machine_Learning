import pandas as pd
import numpy as np

import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.RNN import BasicRNN

from utils.train_model import train
from utils.eval_model import evaluate_flexible
from utils.word_to_index import tokenize
from utils.DataSetForDataLoader import TextDataset

from sklearn.model_selection import train_test_split



def run_rnn(path,criterion, epochs,patience,device,predFunction,printLoss,max_vocab = 5000,max_len = 100):
    data = _load_IMDB_data(path,max_vocab,max_len)
    
    model = BasicRNN(featureDim=1, hiddenDim=128, outputDim=1)
    _train_and_eval_model(model, data, criterion, epochs,patience,device,predFunction,printLoss)
    
    

def _load_IMDB_data(dataPath,max_vocab,max_len):
    df = pd.read_csv(dataPath)  # text,label
    texts = df["text"].values
    labels = df["label"].values

    
    

    encodedTexts, word2idx = tokenize(texts, max_vocab, max_len)
    encodedTextsArray = np.array(encodedTexts, dtype=np.float32)
    encodedTexts3d = encodedTextsArray[:, :, None]
    xTrain, xTest, yTrain, yTest = train_test_split(encodedTexts3d, labels, test_size=0.2)
    trainData = TextDataset(xTrain, yTrain, torch.float32, torch.float32)
    testData = TextDataset(xTest, yTest, torch.float32, torch.float32)
    trainDataLoader = DataLoader(trainData, batch_size=32, shuffle=True)
    testDataLoader = DataLoader(testData, batch_size=32)
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