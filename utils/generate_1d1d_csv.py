import torch
import numpy as np
import pandas as pd


def generate_1d1d_csv(trainSize=1000, testSize=200, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xTrain = torch.linspace(0, 1, trainSize)
    yNoice = torch.randn(trainSize) * 0.1
    yTrain = torch.sin(2 * np.pi * xTrain) + 0.5 * torch.cos(5 * np.pi * xTrain) + 0.3 * xTrain**2 + yNoice
    
    xTest = torch.linspace(0.05, 0.95, testSize)
    yNoice_test = torch.randn(testSize) * 0.1
    yTest = torch.sin(2 * np.pi * xTest) + 0.5 * torch.cos(5 * np.pi * xTest) + 0.3 * xTest**2 + yNoice_test
    
    return {
        'xTrain': xTrain,
        'yTrain': yTrain,
        'xTest': xTest,
        'yTest': yTest
    }
    
    