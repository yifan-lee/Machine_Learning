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

mod = 'word_embedding'

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
    
if mod == 'word_embedding':
    from run.run_word_bedding import run_word_embedding
    
    dataPath = r'./data/word_embedding/text8'
    embeddingsPath = r'./output/word_embedding/embeddings.csv'
    encodePath = r'./output/word_embedding/encoder.csv'
    countOfWordEmbeded = 5000 
    sentenceLength = 100
    embeddingDim = 100
    trainSize = 1000000
    criterion = nn.CrossEntropyLoss()
    epochs=100
    patience = 1
    modelTrained, word2idx = run_word_embedding(
        path = dataPath,
        embeddingDim = embeddingDim,
        countOfWordEmbeded=countOfWordEmbeded,
        sentenceLength = sentenceLength,
        criterion = criterion, 
        trainSize = trainSize,
        epochs=epochs,
        patience = patience,
        device=device,
        printLoss = True
    )
    embeddings = modelTrained.get_embeddings()
    pd.DataFrame(embeddings).to_csv(embeddingsPath)
    pd.DataFrame(list(word2idx.items()), columns=["Word", "encode"]).to_csv(encodePath)
print("Success!")
