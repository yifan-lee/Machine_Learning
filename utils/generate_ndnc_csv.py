import torch
import numpy as np
import pandas as pd


def generate_ndnc_csv(inputDim=3, outputClass=10, trainSize=1000, testSize=200, seed = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    def target_function(x,i):
        return(
            np.sin((2+i) * np.pi * x) + 
            0.5 * np.cos((5+i) * np.pi * x) + 
            0.2 * np.sin((10+i) * np.pi * x + 0.6) +
            0.3 * x**2
        )
        
    xTrain = torch.randn(trainSize, inputDim)
    xTest = torch.randn(testSize, inputDim)

    resultTrain = []
    resultTest = []
    for i in range(0, outputClass):
        features = xTrain.sum(dim=1)
        yTrainI = target_function(features, i)
        yTrainI += 0.1 * torch.randn_like(features)
        resultTrain.append(yTrainI)

        yTestI = target_function(xTest.sum(dim=1), i)
        yTestI += 0.1 * torch.randn_like(xTest.sum(dim=1))
        resultTest.append(yTestI)
    
    yTrainNumeric = torch.stack(resultTrain, dim=0).T
    yTestNumeric = torch.stack(resultTest, dim=0).T

    sortTrain = torch.argsort(yTrainNumeric, dim=1)
    sortTest = torch.argsort(yTestNumeric, dim=1)
    midPos = yTrainNumeric.shape[1] // 2
    yTrainClass = sortTrain[:, midPos]
    yTestClass = sortTest[:, midPos]
    
    return {
        'xTrain': xTrain,
        'yTrain': yTrainClass,
        'xTest': xTest,
        'yTest': yTestClass
    }

if __name__ == "__main__":
    simulationData = generate_ndnc_csv(inputDim=3, outputClass=10, trainSize=1000, testSize=200, seed=0)  
    
    print('success')