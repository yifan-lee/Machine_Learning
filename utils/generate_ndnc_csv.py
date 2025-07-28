import torch
import numpy as np
import pandas as pd


def generate_ndnc_csv(inputDim=3, outputClass=10, trainSize=1000, testSize=200, seed = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    xTrain = torch.randn(trainSize, inputDim)
    xTest = torch.randn(testSize, inputDim)

    resultTrain = []
    resultTest = []
    for i in range(0, outputClass):
        features = xTrain.sum(dim=1)
        yTrainI = torch.sin((2+i)*np.pi*features) + torch.cos((5+i)*np.pi*features) + 0.3*features ** 2
        yTrainI += 0.1 * torch.randn_like(features)
        resultTrain.append(yTrainI)

        yTestI = torch.sin((2+i)*np.pi*xTest.sum(dim=1)) + torch.cos((5+i)*np.pi*xTest.sum(dim=1)) + 0.3*xTest.sum(dim=1) ** 2
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