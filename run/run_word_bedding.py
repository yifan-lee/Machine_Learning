import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam

from sklearn.model_selection import train_test_split

from utils.word_to_index import tokenize
from utils.DataSetForDataLoader import numericDataset

from utils.train_model import train

from model.word_embedding import WordEmbedding



def run_word_embedding(
    path, 
    embeddingDim,
    countOfWordEmbeded, 
    sentenceLength, 
    trainSize,
    criterion, 
    epochs,
    patience,
    device,
    printLoss
):
    text = _load_text8_data(path, countOfWordEmbeded, sentenceLength)
    encoded, word2idx = tokenize(text)
    pairsDf = _get_pairs_from_encoded(encoded,pairLen = 2)
    pairsDfSub = pairsDf[:trainSize]
    data = _transfer_xy_to_DataLoader(pairsDfSub['x'].values,pairsDfSub['y'].values,xType=torch.int64, yType=torch.int64,test_size=0.2,batch_size=32)
    model = WordEmbedding(countOfWordEmbeded, embeddingDim)
    modelTrained = _train_model(
        model, 
        data, 
        criterion, 
        epochs,
        patience,
        device,
        printLoss
    )
    return modelTrained, word2idx
    
    
def _load_text8_data(dataPath, countOfWordEmbeded = 5000, sentenceLength = 100):
    with open(dataPath, "r") as f:
        text = f.read()
    tokens = text.lower().split()
    text = [" ".join(tokens[i:(i+sentenceLength)]) for i in range(0, len(tokens), sentenceLength)]
    return text

def _get_pairs_from_encoded(encoded,pairLen = 2):
    pairs = []
    for sentence in encoded:
        for i, ids in enumerate(sentence):
            sentenceLength = len(sentence)
            for j in range(max(0, i-pairLen), min(sentenceLength, i+pairLen+1)):
                if j == i:
                    pass
                else:
                    pairs.append((ids, sentence[j]))
    pairsDf = pd.DataFrame(pairs, columns=["x", "y"])
    return pairsDf


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


def _train_model(model, data, criterion, epochs,patience,device,printLoss):
    trainCNNLoader = data['trainDataLoader']
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
        printLoss=printLoss,
    )
    return modelTrained