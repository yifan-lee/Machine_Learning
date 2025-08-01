import pandas as pd

def load_data_from_csv(path):
    xTrain = pd.read_csv(path['xTrain'], header=None).values
    yTrain = pd.read_csv(path['yTrain'], header=None).values
    xTest = pd.read_csv(path['xTest'], header=None).values
    yTest = pd.read_csv(path['yTest'], header=None).values
    dataRaw = {
        'xTrain': xTrain,
        'yTrain': yTrain,
        'xTest': xTest,
        'yTest': yTest
    }
    return dataRaw