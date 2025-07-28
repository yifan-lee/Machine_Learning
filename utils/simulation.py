import torch
import numpy as np
import pandas as pd

from combine_train_test_data import combine_train_test_data
from generate_1d1d_csv import generate_1d1d_csv
from generate_ndnc_csv import generate_ndnc_csv





if __name__ == "__main__":
    simulationData = generate_1d1d_csv(trainSize=1000, testSize=200, seed=0)

    xTrain = pd.DataFrame(simulationData['xTrain'].numpy())
    yTrain = pd.DataFrame(simulationData['yTrain'].numpy())
    xTest = pd.DataFrame(simulationData['xTest'].numpy())
    yTest = pd.DataFrame(simulationData['yTest'].numpy())

    xTrain.to_csv('./data/x_train_1d1d.csv', index=False)
    yTrain.to_csv('./data/y_train_1d1d.csv', index=False)
    xTest.to_csv('./data/x_test_1d1d.csv', index=False)
    yTest.to_csv('./data/y_test_1d1d.csv', index=False)
    print("Save 1d1d")
    
    inputDim = 3
    outputClass = 10
    simulationData = generate_ndnc_csv(inputDim=inputDim, outputClass=outputClass, trainSize=1000, testSize=200, seed=0)
    xTrain = pd.DataFrame(simulationData['xTrain'].numpy())
    yTrain = pd.DataFrame(simulationData['yTrain'].numpy())
    xTest = pd.DataFrame(simulationData['xTest'].numpy())
    yTest = pd.DataFrame(simulationData['yTest'].numpy())
    xTrain.to_csv('./data/x_train_ndnc.csv', index=False)
    yTrain.to_csv('./data/y_train_ndnc.csv', index=False)
    xTest.to_csv('./data/x_test_ndnc.csv', index=False)
    yTest.to_csv('./data/y_test_ndnc.csv', index=False)

    print("Save ndnc")

