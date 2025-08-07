import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from utils.word_to_index import tokenize
from utils.DataSetForDataLoader import numericDataset



def run_word_embedding(dataPath, wordCountUpperBound = 5000, sentenceLength = 100):
    text = _load_text8_data(dataPath, wordCountUpperBound = 5000, sentenceLength = 100)
    encoded, word2idx = tokenize(text)
    pairsDf = _get_pairs_from_encoded(encoded,pairLen = 2)
    data = _transfer_xy_to_DataLoader(pairsDf['x'],pairsDf['y'],xType=torch.int64, yType=torch.int64,test_size=0.2,batch_size=32)

def _load_text8_data(dataPath, wordCountUpperBound = 5000, sentenceLength = 100):
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