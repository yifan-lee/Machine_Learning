import torch
import numpy as np
import pandas as pd


def generate_1d1d_csv(trainSize=1000, testSize=200, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    def target_function(x):
        return(
            np.sin(2 * np.pi * x) + 
            0.5 * np.cos(5 * np.pi * x) + 
            0.2 * np.sin(10 * np.pi * x + 0.6) +
            0.3 * x**2
        )
        
    xTrain = torch.linspace(0, 1, trainSize)
    yNoice = torch.randn(trainSize) * 0.1
    yTrain = target_function(xTrain) + yNoice

    xTest = torch.linspace(0.05, 0.95, testSize)
    yNoice_test = torch.randn(testSize) * 0.1
    yTest = target_function(xTest) + yNoice_test

    return {
        'xTrain': xTrain,
        'yTrain': yTrain,
        'xTest': xTest,
        'yTest': yTest
    }
    
    