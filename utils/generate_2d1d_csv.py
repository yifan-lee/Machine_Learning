import torch
import numpy as np
import pandas as pd


def generate_2d1d_csv(trainSize=1000, testSize=200, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    def target_function(x1, x2):
        return(
            np.sin(2 * np.pi * x1) + 
            0.5 * np.cos(5 * np.pi * x1) + 
            0.2 * np.sin(10 * np.pi * x1 + 0.6) +
            0.3 * x1**2 +
            np.sin(14.7 * np.pi * x2) + 
            0.21 * np.cos(25.8 * np.pi * x2) + 
            0.57 * np.sin(10 * np.pi * x2 + 0.6) +
            0.22 * x2**2 +
            np.sin(0.3 * np.pi * x1 * x2) + 
            0.13 * x1 * x2
        )
        
    xTrain = torch.randn(trainSize, 2)
    yNoice = torch.randn(trainSize) * 0.1
    yTrain = target_function(xTrain[:, 0], xTrain[:, 1]) + yNoice
    
    xTest = torch.randn(testSize, 2)
    yNoice_test = torch.randn(testSize) * 0.1
    yTest = target_function(xTest[:, 0], xTest[:, 1]) + yNoice_test

    return {
        'xTrain': xTrain,
        'yTrain': yTrain,
        'xTest': xTest,
        'yTest': yTest
    }
    
    