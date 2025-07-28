import torch
import numpy as np
import pandas as pd

from combine_train_test_data import combine_train_test_data
from generate_1d1d_csv import generate_1d1d_csv
from generate_ndnc_csv import generate_ndnc_csv





if __name__ == "__main__":
    simulationData = generate_1d1d_csv(trainSize=1000, testSize=200, seed=0)

    XcombinedDf = combine_train_test_data(simulationData['xTrain'], simulationData['xTest'])
    YcombinedDf = combine_train_test_data(simulationData['yTrain'], simulationData['yTest'])
    
    

    XcombinedDf.to_csv('./data/x_1d1d.csv', index=False)
    YcombinedDf.to_csv('./data/y_1d1d.csv', index=False)
    print("CSV x_1d1d.csv 和 y_1d1d.csv")
    
    # simulationData = generate_ndnc_csv(inputDim=3, outputClass=10, trainSize=1000, testSize=200, seed=0)
    # trainDf = pd.DataFrame({'x': simulationData['xTrain'].numpy(), 'y': simulationData['yTrain'].numpy()})
    # testDf = pd.DataFrame({'x': simulationData['xTest'].numpy(), 'y': simulationData['yTest'].numpy()})
    
    # trainDf.to_csv('./data/train_ndnc.csv', index=False)
    # testDf.to_csv('./data/test_ndnc.csv', index=False)
    # print("CSV train_ndnc.csv 和 test_ndnc.csv")

